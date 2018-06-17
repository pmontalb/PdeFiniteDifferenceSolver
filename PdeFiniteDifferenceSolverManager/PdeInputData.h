#pragma once

#include <Vector.h>
#include <ColumnWiseMatrix.h>
#include <Tensor.h>
#include <FiniteDifferenceTypes.h>

namespace pde
{
	/**
	*	Supports up to 3D input data.
	*/
	template<typename BcType, MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class PdeInputData
	{
	public:
		cl::Tensor<memorySpace, mathDomain> initialCondition;

		/**
		* Space discretization mesh
		*/
		cl::Tensor<memorySpace, mathDomain> spaceGrid;

		/**
		* Advection coefficient
		*/
		cl::Tensor<memorySpace, mathDomain> velocity;

		/**
		* Diffusion coefficient
		*/
		cl::Tensor<memorySpace, mathDomain> diffusion;

		/**
		* Time discretization mesh size
		*/
		const double dt;

		/**
		* Solver Type
		*/
		const SolverType solverType;

		/**
		* Space Discretizer Type
		*/
		const SpaceDiscretizerType spaceDiscretizerType;

		PdeInputData(const cl::Tensor<memorySpace, mathDomain>& initialCondition,
						const cl::Tensor<memorySpace, mathDomain>& spaceGrid,
						const cl::Tensor<memorySpace, mathDomain>& velocity,
						const cl::Tensor<memorySpace, mathDomain>& diffusion,
						const double dt,
						const SolverType solverType,
					    const SpaceDiscretizerType spaceDiscretizerType)
			: initialCondition(initialCondition),
			spaceGrid(spaceGrid),
			velocity(velocity),
			diffusion(diffusion),
			dt(dt),
			solverType(solverType),
			spaceDiscretizerType(spaceDiscretizerType)
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
	};
}
