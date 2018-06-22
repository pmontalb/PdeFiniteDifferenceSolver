#pragma once

#include <FiniteDifferenceSolver1D.h>

#define MAKE_DEFAULT_CONSTRUCTORS(CLASS)\
	virtual ~CLASS() noexcept = default;\
	CLASS(const CLASS& rhs) noexcept = default;\
	CLASS(CLASS&& rhs) noexcept = default;\
	CLASS& operator=(const CLASS& rhs) noexcept = default;\
	CLASS& operator=(CLASS&& rhs) noexcept = default;

namespace pde
{
	template<MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class WaveEquationSolver1D : public FiniteDifferenceSolver1D<WaveEquationSolver1D<memorySpace, mathDomain>, memorySpace, mathDomain>
	{
	public:
		// befriend the grandparent CRTP class
		friend class FiniteDifferenceSolver<WaveEquationSolver1D<memorySpace, mathDomain>, PdeInputData1D<memorySpace, mathDomain>, memorySpace, mathDomain>;
		// befriend the mother CRTP class
		friend class FiniteDifferenceSolver1D<WaveEquationSolver1D<memorySpace, mathDomain>, memorySpace, mathDomain>;

		using FiniteDifferenceSolver1D::FiniteDifferenceSolver1D;

		MAKE_DEFAULT_CONSTRUCTORS(WaveEquationSolver1D);

	protected:
		void AdvanceImpl(cl::ColumnWiseMatrix<memorySpace, mathDomain>& solution,
						 const std::shared_ptr<cl::Tensor<memorySpace, mathDomain>>& timeDiscretizers,
						 const SolverType solverType,
						 const unsigned nSteps = 1)
		{
			// NB: I am not writing support for multi-step algorithm here, so I will write the Iterate code here in place

			FiniteDifferenceInput1D _input(inputData.dt,
										   inputData.spaceGrid.GetBuffer(),
										   inputData.velocity.GetBuffer(),
										   inputData.diffusion.GetBuffer(),
										   solverType,
										   inputData.spaceDiscretizerType,
										   inputData.boundaryConditions);

			cl::ColumnWiseMatrix<memorySpace, mathDomain> solutionBuffer(solution), solutionDerivativeBuffer(*solutionDerivative);
			cl::ColumnWiseMatrix<memorySpace, mathDomain> workBuffer(solution);
			
			// for performance reasons, I'm creating two working buffers here, which I will re-use during the main loop
			cl::ColumnWiseMatrix<memorySpace, mathDomain> *inSol = &solution, *outSol = &solutionBuffer;
			cl::ColumnWiseMatrix<memorySpace, mathDomain> *inDer = solutionDerivative.get(), *outDer = &solutionDerivativeBuffer;
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
				pde::detail::SetBoundaryConditions1D(outSol->GetBuffer(), _input);
				// outSol = u_{n + 1} = A * (u_n + dt * v_n)

				// v' = L * u ==> v_{n + 1} = A * (v_n + dt * L * u_n)
				cl::Multiply(workBuffer, *spaceDiscretizer, *inSol, MatrixOperation::None, MatrixOperation::None, inputData.dt);  // w = L * u_n * dt
				workBuffer.AddEqual(*inDer);  // w = v_n + dt * L * u_n
				cl::Multiply(*outDer, *timeDiscretizers->matrices[0], workBuffer);
				pde::detail::SetBoundaryConditions1D(outDer->GetBuffer(), _input);
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

		void MakeTimeDiscretizer(const std::shared_ptr<cl::Tensor<memorySpace, mathDomain>>& timeDiscretizers, const SolverType solverType)
		{
			// reset everything to 0
			spaceDiscretizer = std::make_shared<cl::ColumnWiseMatrix<memorySpace, mathDomain>>(solution->nRows(), solution->nRows(), 0.0);
			timeDiscretizers->Set(0.0);

			// since u_xx is multiplied by velocity^2, there's no actual component for u_x
			cl::Vector<memorySpace, mathDomain> velocity(inputData.velocity.size(), static_cast<cl::Vector<memorySpace, mathDomain>::stdType>(0.0));

			// since u_xx is multiplied by velocity^2, the 'diffusion' component is velocity^2
			cl::Vector<memorySpace, mathDomain> diffusion(*(inputData.velocity.matrices[0]->columns[0]));
			diffusion %= diffusion;

			FiniteDifferenceInput1D _input(inputData.dt,
										   inputData.spaceGrid.GetBuffer(),
										   velocity.GetBuffer(),
										   diffusion.GetBuffer(),
										   solverType,
										   inputData.spaceDiscretizerType,
										   inputData.boundaryConditions);
			pde::detail::MakeSpaceDiscretizer1D(spaceDiscretizer->GetTile(), _input);
			pde::detail::MakeTimeDiscretizerWaveEquation(timeDiscretizers->GetCube(), spaceDiscretizer->GetTile(), solverType, inputData.dt);
		}

		void Setup(const unsigned solverSteps)
		{
			assert(solverSteps == 1);

			FiniteDifferenceSolver1D<WaveEquationSolver1D<memorySpace, mathDomain>, memorySpace, mathDomain>::Setup(solverSteps);
			// TODO: setup is called before this is properly initialised
			// TODO: read from input instead of setting it to 0
			solutionDerivative = std::make_shared<cl::ColumnWiseMatrix<memorySpace, mathDomain>>(this->inputData.initialCondition.nRows(), solverSteps, static_cast<cl::ColumnWiseMatrix<memorySpace, mathDomain>::stdType>(0.0));
		}
	};

#pragma region Type aliases

	typedef WaveEquationSolver1D<MemorySpace::Device, MathDomain::Float> GpuSingleWaveEquationSolver1D;
	typedef GpuSingleWaveEquationSolver1D GpuFloatWaveEquationSolver1D;
	typedef WaveEquationSolver1D<MemorySpace::Device, MathDomain::Double> GpuDoubleWaveEquationSolver1D;
	typedef WaveEquationSolver1D<MemorySpace::Host, MathDomain::Float> CpuSingleWaveEquationSolver1D;
	typedef CpuSingleWaveEquationSolver1D CpuFloatWaveEquationSolver1D;
	typedef WaveEquationSolver1D<MemorySpace::Host, MathDomain::Double> CpuDoubleSolver1D;
	typedef GpuSingleWaveEquationSolver1D wave1D;
	typedef GpuDoubleWaveEquationSolver1D dwave1D;

#pragma endregion
}

#undef MAKE_DEFAULT_CONSTRUCTORS

