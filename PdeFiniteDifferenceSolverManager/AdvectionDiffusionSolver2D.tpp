#pragma once

#include <AdvectionDiffusionSolver2D.h>

namespace pde
{
	template<MemorySpace ms, MathDomain md>
	void AdvectionDiffusionSolver2D<ms, md>::MakeTimeDiscretizer(const std::shared_ptr<cl::Tensor<ms, md>>& timeDiscretizers, const SolverType solverType)
	{
		// reset everything to 0
		const unsigned dimension = this->inputData.initialCondition.nRows() * this->inputData.initialCondition.nCols();
        this->spaceDiscretizer = std::make_shared<cl::ColumnWiseMatrix<ms, md>>(dimension, dimension, 0.0);
        timeDiscretizers->Set(0.0);

		FiniteDifferenceInput2D _input(this->inputData.dt,
                                       this->inputData.xSpaceGrid.GetBuffer(),
                                       this->inputData.ySpaceGrid.GetBuffer(),
                                       this->inputData.xVelocity.GetBuffer(),
                                       this->inputData.yVelocity.GetBuffer(),
                                       this->inputData.diffusion.GetBuffer(),
									   solverType,
                                       this->inputData.spaceDiscretizerType,
                                       this->inputData.boundaryConditions);
		pde::detail::MakeSpaceDiscretizer2D(this->spaceDiscretizer->GetTile(), _input);
		pde::detail::MakeTimeDiscretizerAdvectionDiffusion(timeDiscretizers->GetCube(), this->spaceDiscretizer->GetTile(), solverType, this->inputData.dt);
	}
}

