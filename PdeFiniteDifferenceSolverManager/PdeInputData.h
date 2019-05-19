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
					 const double dt,
					 const SolverType solverType,
					 const SpaceDiscretizerType spaceDiscretizerType)
			: initialCondition(initialCondition),
			dt(dt),
			solverType(solverType),
			spaceDiscretizerType(spaceDiscretizerType)
		{
		}

		PdeInputData(const cl::Tensor<memorySpace, mathDomain>& initialCondition,
					 const double dt,
					 const SolverType solverType)
			: initialCondition(initialCondition),
			dt(dt),
			solverType(solverType)
		{
		}

		virtual ~PdeInputData() noexcept = default;
		PdeInputData(const PdeInputData& rhs) = default;
		PdeInputData(PdeInputData&& rhs) = default;
		PdeInputData& operator=(const PdeInputData& rhs) = default;
		PdeInputData& operator=(PdeInputData&& rhs) = default;
	};
}
