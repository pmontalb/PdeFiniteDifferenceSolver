#pragma once

#include <FiniteDifferenceSolver2D.h>
#include <Exception.h>

namespace pde
{
	template<MemorySpace ms, MathDomain md>
	void FiniteDifferenceSolver2D<ms, md>::MakeTimeDiscretizer(const std::shared_ptr<cl::Tensor<ms, md>>& timeDiscretizers, const SolverType solverType)
	{
		throw NotImplementedException();
	}

	template<MemorySpace ms, MathDomain md>
	void FiniteDifferenceSolver2D<ms, md>::AdvanceImpl(const MemoryTile& solutionTile,
											   const std::shared_ptr<cl::Tensor<ms, md>>& timeDiscretizers,
											   const SolverType solverType,
											   const unsigned nSteps = 1)
	{
		throw NotImplementedException();
	}

	template<MemorySpace ms, MathDomain md>
	void FiniteDifferenceSolver2D<ms, md>::Setup(const unsigned solverSteps)
	{
		const unsigned dimension = this->inputData.initialCondition.nRows(), this->inputData.initialCondition.nCols();

		this->solution = std::make_shared<cl::ColumnWiseMatrix<ms, md>>(dimension * dimension, solverSteps);
		this->solution->Set(this->inputData.initialCondition.matrices[0], 0);
		this->timeDiscretizers = std::make_shared<cl::Tensor<ms, md>>(dimension, dimension, solverSteps);
	}
}
