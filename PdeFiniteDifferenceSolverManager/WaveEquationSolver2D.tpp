#pragma once

#include <WaveEquationSolver2D.h>

namespace pde
{
	template<MemorySpace ms, MathDomain md>
	void WaveEquationSolver2D<ms, md>::AdvanceImpl(cl::ColumnWiseMatrix<ms, md>& solution,
												   const std::shared_ptr<cl::Tensor<ms, md>>& timeDiscretizers,
												   const SolverType solverType,
												   const unsigned nSteps = 1)
	{
		// NB: I am not writing support for multi-step algorithm here, so I will write the Iterate code here in place

		FiniteDifferenceInput2D _input(inputData.dt,
									   inputData.xSpaceGrid.GetBuffer(),
									   inputData.ySpaceGrid.GetBuffer(),
									   inputData.diffusion.GetBuffer(),
									   inputData.diffusion.GetBuffer(),
									   inputData.diffusion.GetBuffer(),
									   solverType,
									   inputData.spaceDiscretizerType,
									   inputData.boundaryConditions);

		cl::ColumnWiseMatrix<ms, md> solutionBuffer(solution), solutionDerivativeBuffer(*solutionDerivative);
		cl::ColumnWiseMatrix<ms, md> workBuffer(solution);

		// for performance reasons, I'm creating two working buffers here, which I will re-use during the main loop
		cl::ColumnWiseMatrix<ms, md> *inSol = &solution, *outSol = &solutionBuffer;
		cl::ColumnWiseMatrix<ms, md> *inDer = solutionDerivative.get(), *outDer = &solutionDerivativeBuffer;
		bool needToCopyBack = false;

		// u'' = L * u  
		//		==>
		// u' = v; v' = L * u
		for (unsigned n = 0; n < nSteps; ++n)
		{
			// u' = v ==> u_{n + 1} = A * (u_n + dt * v_n)
			workBuffer.ReadFrom(*inSol);  // w = u_n
			workBuffer.AddEqual(*inDer, inputData.dt);  // w = u_n + dt * v_n
			cl::Multiply(*outSol, *timeDiscretizers->matrices[0], workBuffer);
			pde::detail::SetBoundaryConditions2D(outSol->GetBuffer(), _input);
			// outSol = u_{n + 1} = A * (u_n + dt * v_n)

			// v' = L * u ==> v_{n + 1} = A * (v_n + dt * L * u_n)
			cl::Multiply(workBuffer, *spaceDiscretizer, *inSol, MatrixOperation::None, MatrixOperation::None, inputData.dt);  // w = L * u_n * dt
			workBuffer.AddEqual(*inDer);  // w = v_n + dt * L * u_n
			cl::Multiply(*outDer, *timeDiscretizers->matrices[0], workBuffer);
			pde::detail::SetBoundaryConditions2D(outDer->GetBuffer(), _input);
			// outDer = v_{n + 1} = A * (v_n + dt * L * u_n)

			std::swap(inSol, outSol);
			std::swap(inDer, outDer);
			needToCopyBack = !needToCopyBack;
		}

		if (needToCopyBack)
		{
			solution.ReadFrom(*inSol);
			solutionDerivative->ReadFrom(*inDer);
		}
	}

	template<MemorySpace ms, MathDomain md>
	void WaveEquationSolver2D<ms, md>::MakeTimeDiscretizer(const std::shared_ptr<cl::Tensor<ms, md>>& timeDiscretizers, const SolverType solverType)
	{
		// reset everything to 0
		const unsigned dimension = this->inputData.initialCondition.nRows() * this->inputData.initialCondition.nCols();
		spaceDiscretizer = std::make_shared<cl::ColumnWiseMatrix<ms, md>>(dimension, dimension, 0.0);
		timeDiscretizers->Set(0.0);

		// since u_xx is multiplied by velocity^2, there's no actual component for u_x
		cl::Vector<ms, md> velocity(inputData.xVelocity.size(), static_cast<cl::Vector<ms, md>::stdType>(0.0));

		// since u_xx is multiplied by velocity^2, the 'diffusion' component is velocity^2
		// not really ideal, but I'm reading it from xVelocity, pretty much arbitrarily
		auto _v = inputData.xVelocity.Get();  // FIXME: temp hack!
		cl::Vector<ms, md> diffusion(dimension, _v[0]);
		diffusion %= diffusion;

		FiniteDifferenceInput2D _input(inputData.dt,
									   inputData.xSpaceGrid.GetBuffer(),
									   inputData.ySpaceGrid.GetBuffer(),
									   velocity.GetBuffer(),
									   velocity.GetBuffer(),
									   diffusion.GetBuffer(),
									   solverType,
									   inputData.spaceDiscretizerType,
									   inputData.boundaryConditions);
		pde::detail::MakeSpaceDiscretizer2D(spaceDiscretizer->GetTile(), _input);
		pde::detail::MakeTimeDiscretizerWaveEquation(timeDiscretizers->GetCube(), spaceDiscretizer->GetTile(), solverType, inputData.dt);
	}

	template<MemorySpace ms, MathDomain md>
	void WaveEquationSolver2D<ms, md>::Setup(const unsigned solverSteps)
	{
		assert(solverSteps == 1);

		FiniteDifferenceSolver2D<WaveEquationSolver2D<ms, md>, ms, md>::Setup(solverSteps);

		// TODO: read from input instead of setting it to 0
		const unsigned dimension = this->inputData.initialCondition.nRows() * this->inputData.initialCondition.nCols();
		solutionDerivative = std::make_shared<cl::ColumnWiseMatrix<ms, md>>(dimension, solverSteps, static_cast<cl::ColumnWiseMatrix<ms, md>::stdType>(0.0));
	}
}
