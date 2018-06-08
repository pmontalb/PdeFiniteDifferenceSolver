#pragma once

#include <Vector.h>
#include <ColumnWiseMatrix.h>
#include <Tensor.h>
#include <FiniteDifferenceTypes.h>

namespace pde
{
	/**
	*	Supports up to 3D input data
	*
	*	CRTP implementation
	*/
	template<typename BcType, MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class PdeInputData
	{
	public:
		const cl::Tensor<memorySpace, mathDomain>& initialCondition;

		/**
		* Space discretization mesh
		*/
		const cl::Tensor<memorySpace, mathDomain>& spaceGrid;

		/**
		* Advection coefficient
		*/
		const cl::Tensor<memorySpace, mathDomain>& velocity;

		/**
		* Diffusion coefficient
		*/
		const cl::Tensor<memorySpace, mathDomain>& diffusion;

		/**
		* Time discretization mesh size
		*/
		const double dt;

		/**
		* Solver Type
		*/
		const SolverType solverType;

		PdeInputData(const cl::Tensor<memorySpace, mathDomain>& initialCondition,
						const cl::Tensor<memorySpace, mathDomain>& spaceGrid,
						const cl::Tensor<memorySpace, mathDomain>& velocity,
						const cl::Tensor<memorySpace, mathDomain>& diffusion,
						const double dt,
						const SolverType solverType)
			: initialCondition(initialCondition),
			spaceGrid(spaceGrid),
			velocity(velocity),
			diffusion(diffusion),
			dt(dt),
			solverType(solverType)
		{
		}

		PdeInputData(const cl::Tensor<memorySpace, mathDomain>& initialCondition,
						const cl::Tensor<memorySpace, mathDomain>& spaceGrid,
						const typename cl::Traits<mathDomain>::stdType velocity,
						const typename cl::Traits<mathDomain>::stdType diffusion,
						const double dt,
						const SolverType solverType)
			: initialCondition(initialCondition),
			spaceGrid(spaceGrid),
			velocity(initialCondition.nRows(), initialCondition.nCols(), initialCondition.nCubes(), velocity),
			diffusion(initialCondition.nRows(), initialCondition.nCols(), initialCondition.nCubes(), diffusion),
			dt(dt),
			solverType(solverType)
		{
		}

		virtual ~PdeInputData() noexcept = default;
		PdeInputData(const PdeInputData& rhs) noexcept = default;
		PdeInputData(PdeInputData&& rhs) noexcept = default;
		PdeInputData& operator=(const PdeInputData& rhs) noexcept = default;
		PdeInputData& operator=(PdeInputData&& rhs) noexcept = default;

	protected:
		// having this an abstract method rather than a concrete one prevents this to be instantiated
		virtual BcType GetBoundaryCondition() const noexcept = 0;
	};

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
					   const BoundaryCondition1D boundaryConditions = BoundaryCondition1D())
			: 
			PdeInputData(initialCondition, spaceGrid,
						 cl::Tensor<memorySpace, mathDomain>(initialCondition.size(), 1, 1, velocity),
						 cl::Tensor<memorySpace, mathDomain>(initialCondition.size(), 1, 1, diffusion),
						 dt,
						 solverType),
			boundaryConditions(boundaryConditions)
		{
		}

	protected:
		BoundaryCondition1D GetBoundaryCondition() const noexcept { return boundaryConditions; };
	};

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
					   const BoundaryCondition2D boundaryConditions = BoundaryCondition2D())
			: initialCondition(initialCondition),
			spaceGrid(spaceGrid),
			velocity(velocity),
			diffusion(diffusion),
			dt(dt),
			solverType(solverType),
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
	protected:
		PdeInputData2D GetBoundaryCondition() const noexcept { return boundaryConditions; };
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
