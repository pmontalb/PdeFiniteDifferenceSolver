#pragma once

#include <ColumnWiseMatrix.h>
#include <FiniteDifferenceTypes.h>
#include <PdeInputData.h>
#include <Tensor.h>
#include <Vector.h>

namespace pde
{
	template<MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class PdeInputData1D: public PdeInputData<BoundaryCondition1D, memorySpace, mathDomain>
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

		PdeInputData1D(const cl::Vector<memorySpace, mathDomain>& initialCondition_, const cl::Vector<memorySpace, mathDomain>& spaceGrid_, const cl::Vector<memorySpace, mathDomain>& velocity_, const cl::Vector<memorySpace, mathDomain>& diffusion_, const typename cl::Vector<memorySpace, mathDomain>::stdType dt_, const SolverType solverType_, const SpaceDiscretizerType spaceDiscretizerType_, const BoundaryCondition1D& boundaryConditions_ = BoundaryCondition1D()) : PdeInputData<BoundaryCondition1D, memorySpace, mathDomain>(cl::Tensor<memorySpace, mathDomain>(initialCondition_), dt_, solverType_, spaceDiscretizerType_), velocity(velocity_), diffusion(diffusion_), spaceGrid(spaceGrid_), boundaryConditions(boundaryConditions_) {}

		PdeInputData1D(const cl::Vector<memorySpace, mathDomain>& initialCondition_, const cl::Vector<memorySpace, mathDomain>& spaceGrid_, const typename cl::Traits<mathDomain>::stdType velocity_, const typename cl::Traits<mathDomain>::stdType diffusion_, const typename cl::Vector<memorySpace, mathDomain>::stdType dt_, const SolverType solverType_, const SpaceDiscretizerType spaceDiscretizerType_, const BoundaryCondition1D& boundaryConditions_ = BoundaryCondition1D()) : PdeInputData<BoundaryCondition1D, memorySpace, mathDomain>(cl::Tensor<memorySpace, mathDomain>(initialCondition_), static_cast<double>(dt_), solverType_, spaceDiscretizerType_), velocity(cl::Vector<memorySpace, mathDomain>(initialCondition_.size(), velocity_)), diffusion(cl::Vector<memorySpace, mathDomain>(initialCondition_.size(), diffusion_)), spaceGrid(spaceGrid_), boundaryConditions(boundaryConditions_) {}

		PdeInputData1D(PdeInputData1D&& rhs) = default;
	};

#pragma region Type aliases

	typedef PdeInputData1D<MemorySpace::Device, MathDomain::Float> GpuSinglePdeInputData1D;
	typedef GpuSinglePdeInputData1D GpuFloatPdeInputData1D;
	typedef PdeInputData1D<MemorySpace::Device, MathDomain::Double> GpuDoublePdeInputData1D;

	typedef PdeInputData1D<MemorySpace::Host, MathDomain::Float> CpuSinglePdeInputData1D;
	typedef CpuSinglePdeInputData1D CpuFloatPdeInputData1D;
	typedef PdeInputData1D<MemorySpace::Host, MathDomain::Double> CpuDoublePdeInputData1D;

#pragma endregion
}	 // namespace pde
