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

		PdeInputData2D(const cl::ColumnWiseMatrix<memorySpace, mathDomain>& initialCondition_,
					   const cl::Vector<memorySpace, mathDomain>& xSpaceGrid_,
					   const cl::Vector<memorySpace, mathDomain>& ySpaceGrid_,
					   const cl::ColumnWiseMatrix<memorySpace, mathDomain>& xVelocity_,
					   const cl::ColumnWiseMatrix<memorySpace, mathDomain>& yVelocity_,
					   const cl::ColumnWiseMatrix<memorySpace, mathDomain>& diffusion_,
					   const typename cl::ColumnWiseMatrix<memorySpace, mathDomain>::stdType dt_,
					   const SolverType solverType_,
					   const SpaceDiscretizerType spaceDiscretizerType_,
					   const BoundaryCondition2D& boundaryConditions_ = BoundaryCondition2D())
			: PdeInputData<BoundaryCondition2D, memorySpace, mathDomain>(cl::Tensor<memorySpace, mathDomain>(initialCondition_),
						   dt_,
						   solverType_,
						   spaceDiscretizerType_),
			xSpaceGrid(xSpaceGrid_),
			ySpaceGrid(ySpaceGrid_),
			xVelocity(xVelocity_),
			yVelocity(yVelocity_),
			diffusion(diffusion_.Flatten()),
			boundaryConditions(boundaryConditions_)
		{
		}

		PdeInputData2D(const cl::ColumnWiseMatrix<memorySpace, mathDomain>& initialCondition_,
					   const cl::Vector<memorySpace, mathDomain>& xSpaceGrid_,
					   const cl::Vector<memorySpace, mathDomain>& ySpaceGrid_,
					   const typename cl::Traits<mathDomain>::stdType xVelocity_,
					   const typename cl::Traits<mathDomain>::stdType yVelocity_,
					   const typename cl::Traits<mathDomain>::stdType diffusion_,
					   const typename cl::ColumnWiseMatrix<memorySpace, mathDomain>::stdType dt_,
					   const SolverType solverType_,
					   const SpaceDiscretizerType spaceDiscretizerType_,
					   const BoundaryCondition2D& boundaryConditions_ = BoundaryCondition2D())
			: PdeInputData<BoundaryCondition2D, memorySpace, mathDomain>(cl::Tensor<memorySpace, mathDomain>(initialCondition_),
						   static_cast<double>(dt_),
						   solverType_,
						   spaceDiscretizerType_),
			xSpaceGrid(xSpaceGrid_),
			ySpaceGrid(ySpaceGrid_),
			xVelocity(cl::Vector<memorySpace, mathDomain>(initialCondition_.nRows(), xVelocity_)),
			yVelocity(cl::Vector<memorySpace, mathDomain>(initialCondition_.nCols(), yVelocity_)),
			diffusion(cl::Vector<memorySpace, mathDomain>(initialCondition_.size(), diffusion_)),
			boundaryConditions(boundaryConditions_)
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
