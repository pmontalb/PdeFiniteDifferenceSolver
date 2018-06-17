#pragma once

#include <Vector.h>
#include <ColumnWiseMatrix.h>
#include <Tensor.h>
#include <FiniteDifferenceTypes.h>
#include <PdeInputData.h>

namespace pde
{
	template<MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class PdeInputData1D : public PdeInputData<BoundaryCondition1D, memorySpace, mathDomain>
	{
	public:
		const BoundaryCondition1D boundaryConditions = BoundaryCondition1D();

		PdeInputData1D(const cl::Vector<memorySpace, mathDomain>& initialCondition,
					   const cl::Vector<memorySpace, mathDomain>& spaceGrid,
					   const cl::Vector<memorySpace, mathDomain>& velocity,
					   const cl::Vector<memorySpace, mathDomain>& diffusion,
					   const double dt,
					   const SolverType solverType,
					   const BoundaryCondition1D boundaryConditions = BoundaryCondition1D())
			: initialCondition(initialCondition),
			spaceGrid(spaceGrid),
			velocity(velocity),
			diffusion(diffusion),
			dt(dt),
			solverType(solverType),
			boundaryConditions(boundaryConditions)
		{
		}

		PdeInputData1D(const cl::Vector<memorySpace, mathDomain>& initialCondition,
					   const cl::Vector<memorySpace, mathDomain>& spaceGrid,
					   const typename cl::Traits<mathDomain>::stdType velocity,
					   const typename cl::Traits<mathDomain>::stdType diffusion,
					   const double dt,
					   const SolverType solverType,
					   const SpaceDiscretizerType spaceDiscretizerType,
					   const BoundaryCondition1D boundaryConditions = BoundaryCondition1D())
			:
			PdeInputData(initialCondition, spaceGrid,
						 cl::Tensor<memorySpace, mathDomain>(initialCondition.size(), 1, 1, velocity),
						 cl::Tensor<memorySpace, mathDomain>(initialCondition.size(), 1, 1, diffusion),
						 dt,
						 solverType,
						 spaceDiscretizerType),
			boundaryConditions(boundaryConditions)
		{
		}
	};

#pragma region Type aliases

	typedef PdeInputData1D<MemorySpace::Device, MathDomain::Float> GpuSinglePdeInputData;
	typedef GpuSinglePdeInputData GpuFloatPdeInputData;
	typedef PdeInputData1D<MemorySpace::Device, MathDomain::Double> GpuDoublePdeInputData;

	typedef PdeInputData1D<MemorySpace::Host, MathDomain::Float> CpuSinglePdeInputData;
	typedef CpuSinglePdeInputData CpuFloatPdeInputData;
	typedef PdeInputData1D<MemorySpace::Host, MathDomain::Double> CpuDoublePdeInputData;

#pragma endregion
}
