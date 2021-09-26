#pragma once

#include <ColumnWiseMatrix.h>
#include <FiniteDifferenceTypes.h>
#include <Tensor.h>
#include <Vector.h>

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

		PdeInputData(const cl::Tensor<memorySpace, mathDomain>& initialCondition_, const double dt_, const SolverType solverType_, const SpaceDiscretizerType spaceDiscretizerType_) : initialCondition(initialCondition_), dt(dt_), solverType(solverType_), spaceDiscretizerType(spaceDiscretizerType_) {}

		PdeInputData(const cl::Tensor<memorySpace, mathDomain>& initialCondition_, const double dt_, const SolverType solverType_) : initialCondition(initialCondition_), dt(dt_), solverType(solverType_) {}

		virtual ~PdeInputData() noexcept = default;
		PdeInputData(const PdeInputData& rhs) = default;
		PdeInputData(PdeInputData&& rhs) = default;
		PdeInputData& operator=(const PdeInputData& rhs) = default;
		PdeInputData& operator=(PdeInputData&& rhs) = default;
	};
}	 // namespace pde
