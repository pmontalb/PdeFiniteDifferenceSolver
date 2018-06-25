#pragma once

#include <AdvectionDiffusionSolver1D.h>

namespace pde
{
	template<MemorySpace ms, MathDomain md>
	void AdvectionDiffusionSolver1D<ms, md>::MakeTimeDiscretizer(const std::shared_ptr<cl::Tensor<ms, md>>& timeDiscretizers, const SolverType solverType)
	{
		// reset everything to 0
		spaceDiscretizer = std::make_shared<cl::ColumnWiseMatrix<ms, md>>(solution->nRows(), solution->nRows(), 0.0);
		timeDiscretizers->Set(0.0);

		FiniteDifferenceInput1D _input(inputData.dt,
										inputData.spaceGrid.GetBuffer(),
										inputData.velocity.GetBuffer(),
										inputData.diffusion.GetBuffer(),
										solverType,
										inputData.spaceDiscretizerType,
										inputData.boundaryConditions);
		pde::detail::MakeSpaceDiscretizer1D(spaceDiscretizer->GetTile(), _input);
		pde::detail::MakeTimeDiscretizerAdvectionDiffusion(timeDiscretizers->GetCube(), spaceDiscretizer->GetTile(), solverType, inputData.dt);
	}
}

