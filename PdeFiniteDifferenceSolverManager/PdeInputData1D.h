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
        /**
        * Advection coefficient
        */
        cl::Vector<memorySpace, mathDomain> velocity;

        /**
        * Diffusion coefficient
        */
        cl::Vector<memorySpace, mathDomain> diffusion;

		/**
		* Space discretization mesh
		*/
		cl::Vector<memorySpace, mathDomain> spaceGrid;

		const BoundaryCondition1D boundaryConditions = BoundaryCondition1D();

		PdeInputData1D(const cl::Vector<memorySpace, mathDomain>& initialCondition,
					   const cl::Vector<memorySpace, mathDomain>& spaceGrid,
					   const cl::Vector<memorySpace, mathDomain>& velocity,
					   const cl::Vector<memorySpace, mathDomain>& diffusion,
					   const double dt,
					   const SolverType solverType,
					   const SpaceDiscretizerType spaceDiscretizerType,
					   const BoundaryCondition1D boundaryConditions = BoundaryCondition1D())
			: PdeInputData<BoundaryCondition1D, memorySpace, mathDomain>(cl::Tensor<memorySpace, mathDomain>(initialCondition),
						   dt,
						   solverType,
						   spaceDiscretizerType),
			velocity(velocity),
			diffusion(diffusion),
			spaceGrid(spaceGrid),
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
                PdeInputData<BoundaryCondition1D, memorySpace, mathDomain>(cl::Tensor<memorySpace, mathDomain>(initialCondition),
						 dt,
						 solverType,
						 spaceDiscretizerType),
			velocity(cl::Vector<memorySpace, mathDomain>(initialCondition.size(), velocity)),
			diffusion(cl::Vector<memorySpace, mathDomain>(initialCondition.size(), diffusion)),
                spaceGrid(spaceGrid),
			boundaryConditions(boundaryConditions)
		{
		}
	};

#pragma region Type aliases

	typedef PdeInputData1D<MemorySpace::Device, MathDomain::Float> GpuSinglePdeInputData1D;
	typedef GpuSinglePdeInputData1D GpuFloatPdeInputData1D;
	typedef PdeInputData1D<MemorySpace::Device, MathDomain::Double> GpuDoublePdeInputData1D;

	typedef PdeInputData1D<MemorySpace::Host, MathDomain::Float> CpuSinglePdeInputData1D;
	typedef CpuSinglePdeInputData1D CpuFloatPdeInputData1D;
	typedef PdeInputData1D<MemorySpace::Host, MathDomain::Double> CpuDoublePdeInputData1D;

#pragma endregion
}
