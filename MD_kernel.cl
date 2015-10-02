#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

uint xyz_to_global_bi(float4 xyz, float bin_length, uint N, uint N2) {
	int4 local_bi = (int4)(0, 0, 0, 0);
	uint global_bi;

//	local_bi = clamp(convert_int4_rtp(xyz / bin_length - 1), 0u, N - 1u);
	local_bi = clamp(convert_int4_rtn(xyz / bin_length), 0, N - 1);

	global_bi = local_bi.x + local_bi.y * N + local_bi.z * N2;

	return global_bi;
}

int4 global_to_local_bi(uint global_bi, uint N, uint N2) {
	int4 local_bi = (int4)(0, 0, 0, 0);
	uint4 converter = (uint4)(1u, N, N2, 1u);

	local_bi = convert_int4_rtn((global_bi / converter) % N);

	return local_bi;
}

__kernel void reset_uint(__global uint *input, uint items)
{
	uint i = get_global_id(0);
	if (i >= items)
	{
		return;
	}
	input[i] = 0u;
}

__kernel void fill_bins(__global float4 *xyz, __global uint *bin_index, __global uint *ion_count,
		float bin_length, uint N, uint ions)
{
	uint i = get_global_id(0);
	if (i >= ions)
	{
		return;
	}

	uint global_bi;

	global_bi = xyz_to_global_bi(xyz[i], bin_length, N, N*N);

	bin_index[i] = global_bi;
	atomic_inc(&ion_count[global_bi]);
}

__kernel void prefix_sum(__global uint *input, __global uint *output, __global uint *block_sum,
		__local uint *block, uint block_size, uint items)
{
	uint gri = get_group_id(0);

	uint gli_min = 2u*get_global_id(0);
	uint gli_max = gli_min + 1u;

	if (gli_max > items)
	{
		return;
	}

	uint loi_min = 2u*get_local_id(0);
	uint loi_max = loi_min + 1u;

	block[loi_min] = input[gli_min];
	block[loi_max] = input[gli_max];
	barrier(CLK_LOCAL_MEM_FENCE);

	uint2 cache = (uint2)(block[0], block[0] + block[1]);

	for (uint stride = 1u; stride < block_size; stride *= 2u)
	{
		if (loi_min >= stride)
		{
			cache.x = block[loi_min - stride] + block[loi_min];
			cache.y = block[loi_max - stride] + block[loi_max];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		block[loi_min] = cache.x;
		block[loi_max] = cache.y;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	block_sum[gri] = block[block_size - 1u];

	output[gli_min] = block[loi_min];
	output[gli_max] = block[loi_max];
}

__kernel void block_sum(__global uint *prefix_sum, __global uint *block_sum,
		uint items)
{
	uint gri = get_group_id(0);

	uint gli_min = 2u*get_global_id(0);
	uint gli_max = gli_min + 1u;

	if (gri == 0u || gli_max > items)
	{
		return;
	}

	uint loi = get_local_id(0);
	__local uint value;

	if (loi == 0u)
	{
		value = block_sum[0];
		for (uint i = 1u; i < gri; i++)
		{
			value += block_sum[i];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	prefix_sum[gli_min] += value;
	prefix_sum[gli_max] += value;
}

__kernel void counting_sort(__global uint *bin_index, __global uint *prefix_sum,
		__global float4 *xyz, __global float4 *vel, __global float *mass,
		__global float4 *xyz_sorted, __global float4 *vel_sorted, __global float *mass_sorted,
		uint ions)
{
	uint i = get_global_id(0);
	if (i >= ions)
	{
		return;
	}

	uint new_index = atomic_dec(&prefix_sum[bin_index[i]]) - 1u;

	xyz_sorted[new_index] = xyz[i];
	vel_sorted[new_index] = vel[i];
	mass_sorted[new_index] = mass[i];
}

__kernel void reindex(__global float4 *xyz, __global float4 *vel, __global float *mass,
		__global float4 *xyz_sorted, __global float4 *vel_sorted, __global float *mass_sorted,
		uint ions)
{
	uint i = get_global_id(0);
	if (i >= ions)
	{
		return;
	}

	xyz[i] = xyz_sorted[i];
	vel[i] = vel_sorted[i];
	mass[i] = mass_sorted[i];
}

__kernel void get_force(__global float4 *xyz, __global float4 *F,
		__global float *U, __global uint *prefix_sum, __global uint *ion_count,
		float bin_length, uint N, float Rc, float eps, float sig, uint ions)
{
	uint i = get_global_id(0);
	if (i >= ions)
	{
		return;
	}

	uint global_bi, neighbour_bi, private_ps, private_ic;
	uint N2 = N*N;

	float R2, invR, SR, F_lj;
	float Ec = eps*(pown(sig/Rc,12) - pown(sig/Rc,6));
	float Rc2 = Rc*Rc;
	float U_i = 0;

	int4 local_bi = (int4)(0, 0, 0, 0);
	int4 bi_step_neg = (int4)(0, 0, 0, 0);
	int4 bi_step_pos = (int4)(0, 0, 0, 0);

	float4 xyz_i = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
	float4 F_i = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
	float4 r = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

	xyz_i = xyz[i];

	global_bi = xyz_to_global_bi(xyz_i, bin_length, N, N2);
	local_bi = global_to_local_bi(global_bi, N, N2);

	bi_step_neg = global_to_local_bi(xyz_to_global_bi(xyz_i - Rc, bin_length, N, N2), N, N2) - local_bi;
	bi_step_pos = global_to_local_bi(xyz_to_global_bi(xyz_i + Rc, bin_length, N, N2), N, N2) - local_bi;

	for (int dz = bi_step_neg.z; dz <= bi_step_pos.z; dz++)
	{
		for (int dy = bi_step_neg.y; dy <= bi_step_pos.y; dy++)
		{
			for (int dx = bi_step_neg.x; dx <= bi_step_pos.x; dx++)
			{
				neighbour_bi = global_bi + dx + dy*N + dz*N2;
				private_ic = ion_count[neighbour_bi];

				if (private_ic == 0u)
				{
					continue;
				}
				private_ps = prefix_sum[neighbour_bi];

				for (uint j = private_ps; j < private_ps + private_ic; j++)
				{
					if (j == i)
					{
						continue;
					}

					r = xyz_i - xyz[j];
					R2 = dot(r, r);

					if (R2 < Rc2)
					{
						invR = 1/sqrt(R2);
						SR = sig*invR;
						SR = SR*SR*SR*SR*SR*SR;

						F_lj = eps*invR*(2.0f*SR*SR - SR);

						U_i += eps*(SR*SR - SR) - Ec;
						F_i += F_lj*invR*r;
					}
				}
			}
		}
	}
	U[i] = 2.0f*U_i;
	F[i] = 24.0f*F_i;
}

__kernel void verlet_part1(__global float4 *xyz, __global float4 *vel, __global float4 *F,
		__global float *mass, float4 box_size, float dt, uint ions)
{
	uint i = get_global_id(0);
	if (i >= ions)
	{
		return;
	}

	float4 low = (float4)(0.0f, 0.0f, 0.0f, -1.0f);

	float4 new_xyz = xyz[i];
	float4 new_vel = vel[i];

	new_vel += 0.5f * dt * F[i]/mass[i];
	new_xyz += dt * vel[i];

	vel[i] = (new_xyz <= low || new_xyz >= box_size) ? -new_vel : new_vel;
	xyz[i] += dt * vel[i];
}

__kernel void verlet_part2(__global float4 *vel, __global float4 *F,
		__global float *mass, __global float *Ek, __global float *T,
		float dt, uint ions)
{
	uint i = get_global_id(0);
	if (i >= ions)
	{
		return;
	}

	float4 new_vel = vel[i];

	new_vel += 0.5f * dt * F[i]/mass[i];

	Ek[i] = 0.5f * mass[i] * dot(new_vel, new_vel);
	vel[i] = new_vel;
}
