#pragma once

#include <AdvectionDiffusionSolver2D.h>

namespace pde
{
	template<MemorySpace ms, MathDomain md>
	void AdvectionDiffusionSolver2D<ms, md>::MakeTimeDiscretizer(const std::shared_ptr<cl::Tensor<ms, md>>& timeDiscretizers, const SolverType solverType)
	{
		// reset everything to 0
		const unsigned dimension = this->inputData.initialCondition.nRows() * this->inputData.initialCondition.nCols();
		spaceDiscretizer = std::make_shared<cl::ColumnWiseMatrix<ms, md>>(dimension, dimension, 0.0);
		timeDiscretizers->Set(0.0);

		FiniteDifferenceInput2D _input(inputData.dt,
									   inputData.xSpaceGrid.GetBuffer(),
									   inputData.ySpaceGrid.GetBuffer(),
									   inputData.xVelocity.GetBuffer(),
									   inputData.yVelocity.GetBuffer(),
									   inputData.diffusion.GetBuffer(),
									   solverType,
									   inputData.spaceDiscretizerType,
									   inputData.boundaryConditions);
		pde::detail::MakeSpaceDiscretizer2D(spaceDiscretizer->GetTile(), _input);
		pde::detail::MakeTimeDiscretizerAdvectionDiffusion(timeDiscretizers->GetCube(), spaceDiscretizer->GetTile(), solverType, inputData.dt);
	}
}

