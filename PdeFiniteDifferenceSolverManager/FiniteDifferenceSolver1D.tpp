#pragma once

#include <FiniteDifferenceSolver1D.h>

namespace pde
{
	template<MemorySpace ms, MathDomain md>
	void FiniteDifferenceSolver1D<ms, md>::MakeTimeDiscretizer(const std::shared_ptr<cl::Tensor<ms, md>>& timeDiscretizers, const SolverType solverType)
	{
		// reset everything to 0
		cl::ColumnWiseMatrix<ms, md> spaceDiscretizer(solution->nRows(), solution->nRows(), 0.0);
		timeDiscretizers->Set(0.0);

		FiniteDifferenceInput1D _input(inputData.dt,
									   inputData.spaceGrid.GetBuffer(),
									   inputData.velocity.GetBuffer(),
									   inputData.diffusion.GetBuffer(),
									   solverType,
									   inputData.boundaryConditions);
		pde::detail::MakeSpaceDiscretizer1D(spaceDiscretizer.GetTile(), _input);
		pde::detail::MakeTimeDiscretizer1D(timeDiscretizers->GetCube(), spaceDiscretizer.GetTile(), _input);
	}

	template<MemorySpace ms, MathDomain md>
	void FiniteDifferenceSolver1D<ms, md>::AdvanceImpl(const MemoryTile& solutionTile,
											   const std::shared_ptr<cl::Tensor<ms, md>>& timeDiscretizers,
											   const SolverType solverType,
											   const unsigned nSteps)
	{
		FiniteDifferenceInput1D _input(inputData.dt,
									   inputData.spaceGrid.GetBuffer(),
									   inputData.velocity.GetBuffer(),
									   inputData.diffusion.GetBuffer(),
									   solverType,
									   inputData.boundaryConditions);
		pde::detail::Iterate1D(solutionTile, timeDiscretizers->GetCube(), _input, nSteps);
	}

	template<MemorySpace ms, MathDomain md>
	void FiniteDifferenceSolver1D<ms, md>::Setup(const unsigned solverSteps)
	{
		this->solution = std::make_shared<cl::ColumnWiseMatrix<ms, md>>(this->inputData.initialCondition.nRows(), solverSteps);
		this->solution->Set(*this->inputData.initialCondition.matrices[0]->columns[0], solverSteps - 1);
		this->timeDiscretizers = std::make_shared<cl::Tensor<ms, md>>(this->inputData.initialCondition.nRows(), this->inputData.initialCondition.nRows(), solverSteps);

		// need to calculate solution for all the steps > 1
		for (int step = solverSteps - 2; step >= 0; --step)
		{
			// make a volatile CrankNicolson scheme for filling the required steps in the solution
			// WARNING: if the multi-step method is higher than second order, this might reduce the overall accuracy
			constexpr SolverType multiStepEvolutionScheme = { SolverType::CrankNicolson };

			auto tmp = std::make_shared<cl::Tensor<ms, md>>(this->inputData.initialCondition.nRows(), this->inputData.initialCondition.nRows(), 1);
			MakeTimeDiscretizer(tmp, multiStepEvolutionScheme);

			// copy the previous step solution
			this->solution->Set(*solution->columns[step + 1], step);

			// advance with CrankNicolson scheme
			MemoryBuffer _solution;
			extractColumnBufferFromMatrix(_solution, solution->GetTile(), step);
			AdvanceImpl(_solution, tmp, multiStepEvolutionScheme, 1);
		}
	}
}
