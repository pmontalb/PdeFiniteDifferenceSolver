#pragma once

#include <Vector.h>
#include <ColumnWiseMatrix.h>
#include <Tensor.h>
#include <FiniteDifferenceTypes.h>
#include <PdeInputData.h>

namespace pde
{
	template<MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class PdeInputData2D : public PdeInputData<BoundaryCondition2D, memorySpace, mathDomain>
	{
	public:
		const BoundaryCondition2D boundaryConditions = BoundaryCondition2D();

		PdeInputData2D(const cl::ColumnWiseMatrix<memorySpace, mathDomain>& initialCondition,
					   const cl::ColumnWiseMatrix<memorySpace, mathDomain>& spaceGrid,
					   const cl::ColumnWiseMatrix<memorySpace, mathDomain>& velocity,
					   const cl::ColumnWiseMatrix<memorySpace, mathDomain>& diffusion,
					   const double dt,
					   const SolverType solverType,
					   const SpaceDiscretizerType spaceDiscretizerType,
					   const BoundaryCondition2D boundaryConditions = BoundaryCondition2D())
			: PdeInputData(initialCondition, spaceGrid,
						   cl::Tensor<memorySpace, mathDomain>(initialCondition.size(), 1, 1, velocity),
						   cl::Tensor<memorySpace, mathDomain>(initialCondition.size(), 1, 1, diffusion),
						   dt,
						   solverType,
						   spaceDiscretizerType),
			boundaryConditions(boundaryConditions)
		{
		}

		PdeInputData2D(const cl::ColumnWiseMatrix<memorySpace, mathDomain>& initialCondition,
					   const cl::ColumnWiseMatrix<memorySpace, mathDomain>& spaceGrid,
					   const typename cl::Traits<mathDomain>::stdType velocity,
					   const typename cl::Traits<mathDomain>::stdType diffusion,
					   const double dt,
					   const SolverType solverType,
					   const BoundaryCondition2D boundaryConditions = BoundaryCondition2D())
			: initialCondition(initialCondition),
			spaceGrid(spaceGrid),
			velocity(initialCondition.nRows(), initialCondition.nCols(), 1 velocity),
			diffusion(initialCondition.nRows(), initialCondition.nCols(), 1, diffusion),
			dt(dt),
			solverType(solverType),
			boundaryConditions(boundaryConditions)
		{
		}
	};
}
