#pragma once

#include <WaveEquationSolver1D.h>

namespace pde
{
	template<MemorySpace ms, MathDomain md>
	void WaveEquationSolver1D<ms, md>::AdvanceImpl(cl::ColumnWiseMatrix<ms, md>& solution_,
												   const cl::Tensor<ms, md>& timeDiscretizers_,
												   const SolverType solverType,
												   const unsigned nSteps)
	{
		// NB: I am not writing support for multi-step algorithm here, so I will write the Iterate code here in place

		FiniteDifferenceInput1D _input(this->inputData.dt,
                                       this->inputData.spaceGrid.GetBuffer(),
                                       this->inputData.velocity.GetBuffer(),
                                       this->inputData.diffusion.GetBuffer(),
                                       solverType,
                                       this->inputData.spaceDiscretizerType,
                                       this->inputData.boundaryConditions);

		cl::ColumnWiseMatrix<ms, md> solutionBuffer(solution_), solutionDerivativeBuffer(*this->solutionDerivative);
		cl::ColumnWiseMatrix<ms, md> workBuffer(solution_);

		// for performance reasons, I'm creating two working buffers here, which I will re-use during the main loop
		cl::ColumnWiseMatrix<ms, md> *inSol = &solution_, *outSol = &solutionBuffer;
		cl::ColumnWiseMatrix<ms, md> *inDer = this->solutionDerivative.get(), *outDer = &solutionDerivativeBuffer;
		bool needToCopyBack = false;

		// u'' = L * u  
		//		==>
		// u' = v; v' = L * u
		for (unsigned n = 0; n < nSteps; ++n)
		{
			// u' = v ==> u_{n + 1} = A * (u_n + dt * v_n)
			workBuffer.ReadFrom(*inSol);  // w = u_n
			workBuffer.AddEqualMatrix(*inDer, MatrixOperation::None, MatrixOperation::None, 1.0, this->inputData.dt);  // w = u_n + dt * v_n
			timeDiscretizers_.matrices[0]->Multiply(*outSol, workBuffer);
			pde::detail::SetBoundaryConditions1D(outSol->GetTile(), _input);
			// outSol = u_{n + 1} = A * (u_n + dt * v_n)

			// v' = L * u ==> v_{n + 1} = A * (v_n + dt * L * u_n)
			this->spaceDiscretizer->Multiply(workBuffer, *inSol, MatrixOperation::None, MatrixOperation::None, this->inputData.dt);  // w = L * u_n * dt
			workBuffer.AddEqualMatrix(*inDer);  // w = v_n + dt * L * u_n
			this->timeDiscretizers->matrices[0]->Multiply(*outDer, workBuffer);
			pde::detail::SetBoundaryConditions1D(outDer->GetTile(), _input);
			// outDer = v_{n + 1} = A * (v_n + dt * L * u_n)

			std::swap(inSol, outSol);
			std::swap(inDer, outDer);
			needToCopyBack = !needToCopyBack;
		}

		if (needToCopyBack)
		{
            solution_.ReadFrom(*inSol);
            this->solutionDerivative->ReadFrom(*inDer);
		}
	}

	template<MemorySpace ms, MathDomain md>
	void WaveEquationSolver1D<ms, md>::MakeTimeDiscretizer(cl::Tensor<ms, md>& timeDiscretizers_, const SolverType solverType)
	{
		// reset everything to 0
        this->spaceDiscretizer = std::make_unique<cl::ColumnWiseMatrix<ms, md>>(this->solution->nRows(), this->solution->nRows(), 0.0);
		timeDiscretizers_.Set(0.0);

		// since u_xx is multiplied by velocity^2, there's no actual component for u_x
		cl::Vector<ms, md> velocity(this->inputData.velocity.size(), static_cast<typename cl::Vector<ms, md>::stdType>(0.0));

		// since u_xx is multiplied by velocity^2, the 'diffusion' component is velocity^2
		cl::Vector<ms, md> diffusion(this->inputData.velocity);
		diffusion.ElementWiseProduct(diffusion);

		FiniteDifferenceInput1D _input(this->inputData.dt,
                                       this->inputData.spaceGrid.GetBuffer(),
										velocity.GetBuffer(),
										diffusion.GetBuffer(),
										solverType,
                                       this->inputData.spaceDiscretizerType,
                                       this->inputData.boundaryConditions);
		pde::detail::MakeSpaceDiscretizer1D(this->spaceDiscretizer->GetTile(), _input);
		pde::detail::MakeTimeDiscretizerWaveEquation(timeDiscretizers_.GetCube(), this->spaceDiscretizer->GetTile(), solverType, this->inputData.dt);
	}

	template<MemorySpace ms, MathDomain md>
	void WaveEquationSolver1D<ms, md>::Setup(const unsigned solverSteps)
	{
		assert(solverSteps == 1);

		FiniteDifferenceSolver1D<WaveEquationSolver1D<ms, md>, ms, md>::Setup(solverSteps);

		// TODO: read from input instead of setting it to 0
        this->solutionDerivative = std::make_unique<cl::ColumnWiseMatrix<ms, md>>(this->inputData.initialCondition.nRows(), solverSteps, static_cast<typename cl::ColumnWiseMatrix<ms, md>::stdType>(0.0));
	}
}
