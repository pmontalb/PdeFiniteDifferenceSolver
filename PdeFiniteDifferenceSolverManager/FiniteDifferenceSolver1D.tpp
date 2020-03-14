#pragma once

#include <FiniteDifferenceSolver1D.h>

namespace pde
{
	template<class solverImpl, MemorySpace ms, MathDomain md>
	void FiniteDifferenceSolver1D<solverImpl, ms, md>::AdvanceImpl(cl::ColumnWiseMatrix<ms, md>& solution_,
											   const cl::Tensor<ms, md>& timeDiscretizers_,
											   const SolverType solverType,
											   const unsigned nSteps)
	{
		FiniteDifferenceInput1D _input(this->inputData.dt,
                                       this->inputData.spaceGrid.GetBuffer(),
                                       this->inputData.velocity.GetBuffer(),
                                       this->inputData.diffusion.GetBuffer(),
									   solverType,
                                       this->inputData.spaceDiscretizerType,
                                       this->inputData.boundaryConditions);
		pde::detail::Iterate1D(solution_.GetTile(), timeDiscretizers_.GetCube(), _input, nSteps);
	}

	template<class solverImpl, MemorySpace ms, MathDomain md>
	void FiniteDifferenceSolver1D<solverImpl, ms, md>::Setup(const unsigned solverSteps)
	{
        this->solution = std::make_unique<cl::ColumnWiseMatrix<ms, md>>(this->inputData.initialCondition.nRows(), solverSteps);
        this->solution->Set(*this->inputData.initialCondition.matrices[0]->columns[0], solverSteps - 1);
        this->timeDiscretizers = std::make_unique<cl::Tensor<ms, md>>(this->inputData.initialCondition.nRows(), this->inputData.initialCondition.nRows(), solverSteps);

		// need to calculate solution for all the steps > 1
		for (int step = static_cast<int>(solverSteps) - 2; step >= 0; --step)
		{
			// make a volatile CrankNicolson scheme for filling the required steps in the solution
			// WARNING: if the multi-step method is higher than second order, this might reduce the overall accuracy
			constexpr SolverType multiStepEvolutionScheme = { SolverType::CrankNicolson };

			cl::Tensor<ms, md> tmp(this->inputData.initialCondition.nRows(), this->inputData.initialCondition.nRows(), 1);
			static_cast<solverImpl*>(this)->MakeTimeDiscretizer(tmp, multiStepEvolutionScheme);

			// copy the previous step solution
			this->solution->Set(*this->solution->columns[static_cast<unsigned>(step) + 1], static_cast<unsigned>(step));

			// advance with CrankNicolson scheme
			const auto& _solution = this->solution->columns[static_cast<unsigned>(step)];
			FiniteDifferenceInput1D _input(this->inputData.dt,
                                           this->inputData.spaceGrid.GetBuffer(),
                                           this->inputData.velocity.GetBuffer(),
                                           this->inputData.diffusion.GetBuffer(),
										   multiStepEvolutionScheme,
                                           this->inputData.spaceDiscretizerType,
                                           this->inputData.boundaryConditions);
			MemoryTile tmpBuffer(_solution->GetBuffer().pointer, _solution->size(), 1, ms, md);
			pde::detail::Iterate1D(tmpBuffer, tmp.GetCube(), _input, 1);
		}
	}
}
