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
		/**
		* Space discretization mesh
		*/
		cl::Vector<memorySpace, mathDomain> xSpaceGrid;
		cl::Vector<memorySpace, mathDomain> ySpaceGrid;

		/**
		* Advection coefficient
		*/
		cl::Vector<memorySpace, mathDomain> xVelocity;
		cl::Vector<memorySpace, mathDomain> yVelocity;

		/**
		* Diffusion coefficient: depends on both dimension, and here is flattened out
		*/
		cl::Vector<memorySpace, mathDomain> diffusion;

		const BoundaryCondition2D boundaryConditions = BoundaryCondition2D();

		PdeInputData2D(const cl::ColumnWiseMatrix<memorySpace, mathDomain>& initialCondition,
					   const cl::Vector<memorySpace, mathDomain>& xSpaceGrid,
					   const cl::Vector<memorySpace, mathDomain>& ySpaceGrid,
					   const cl::ColumnWiseMatrix<memorySpace, mathDomain>& xVelocity,
					   const cl::ColumnWiseMatrix<memorySpace, mathDomain>& yVelocity,
					   const cl::ColumnWiseMatrix<memorySpace, mathDomain>& diffusion,
					   const double dt,
					   const SolverType solverType,
					   const SpaceDiscretizerType spaceDiscretizerType,
					   const BoundaryCondition2D boundaryConditions = BoundaryCondition2D())
			: PdeInputData(initialCondition,
						   dt,
						   solverType,
						   spaceDiscretizerType),
			xSpaceGrid(xSpaceGrid),
			ySpaceGrid(ySpaceGrid),
			diffusion(diffusion.Flatten()),
			boundaryConditions(boundaryConditions)
		{
		}

		PdeInputData2D(const cl::ColumnWiseMatrix<memorySpace, mathDomain>& initialCondition,
					   const cl::Vector<memorySpace, mathDomain>& xSpaceGrid,
					   const cl::Vector<memorySpace, mathDomain>& ySpaceGrid,
					   const typename cl::Traits<mathDomain>::stdType xVelocity,
					   const typename cl::Traits<mathDomain>::stdType yVelocity,
					   const typename cl::Traits<mathDomain>::stdType diffusion,
					   const double dt,
					   const SolverType solverType,
					   const SpaceDiscretizerType spaceDiscretizerType,
					   const BoundaryCondition2D boundaryConditions = BoundaryCondition2D())
			: PdeInputData(initialCondition,
						   dt,
						   solverType,
						   spaceDiscretizerType),
			xSpaceGrid(xSpaceGrid),
			ySpaceGrid(ySpaceGrid),
			xVelocity(cl::Vector<memorySpace, mathDomain>(initialCondition.nRows(), xVelocity)),
			yVelocity(cl::Vector<memorySpace, mathDomain>(initialCondition.nCols(), yVelocity)),
			diffusion(cl::Vector<memorySpace, mathDomain>(initialCondition.size(), diffusion)),
			boundaryConditions(boundaryConditions)
		{
		}
	};

#pragma region Type aliases

	typedef PdeInputData2D<MemorySpace::Device, MathDomain::Float> GpuSinglePdeInputData2D;
	typedef GpuSinglePdeInputData2D GpuFloatPdeInputData2D;
	typedef PdeInputData2D<MemorySpace::Device, MathDomain::Double> GpuDoublePdeInputData2D;

	typedef PdeInputData2D<MemorySpace::Host, MathDomain::Float> CpuSinglePdeInputData2D;
	typedef CpuSinglePdeInputData2D CpuFloatPdeInputData2D;
	typedef PdeInputData2D<MemorySpace::Host, MathDomain::Double> CpuDoublePdeInputData2D;

#pragma endregion
}
