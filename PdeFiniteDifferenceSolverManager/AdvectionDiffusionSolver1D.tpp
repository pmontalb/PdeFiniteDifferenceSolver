#pragma once

#include <AdvectionDiffusionSolver1D.h>

namespace pde
{
	template<MemorySpace ms, MathDomain md>
	void AdvectionDiffusionSolver1D<ms, md>::MakeTimeDiscretizer(const std::shared_ptr<cl::Tensor<ms, md>>& timeDiscretizers, const SolverType solverType)
	{
		// reset everything to 0
        this->spaceDiscretizer = std::make_shared<cl::ColumnWiseMatrix<ms, md>>(this->solution->nRows(), this->solution->nRows(), 0.0);
        timeDiscretizers->Set(0.0);

		FiniteDifferenceInput1D _input(this->inputData.dt,
                                       this->inputData.spaceGrid.GetBuffer(),
                                       this->inputData.velocity.GetBuffer(),
                                       this->inputData.diffusion.GetBuffer(),
										solverType,
                                       this->inputData.spaceDiscretizerType,
                                       this->inputData.boundaryConditions);
		pde::detail::MakeSpaceDiscretizer1D(this->spaceDiscretizer->GetTile(), _input);
		pde::detail::MakeTimeDiscretizerAdvectionDiffusion(timeDiscretizers->GetCube(), this->spaceDiscretizer->GetTile(), solverType, this->inputData.dt);
	}
}

