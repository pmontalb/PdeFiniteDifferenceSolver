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

		using FiniteDifferenceSolver1D<WaveEquationSolver1D<memorySpace, mathDomain>, memorySpace, mathDomain>::FiniteDifferenceSolver1D;

		MAKE_DEFAULT_CONSTRUCTORS(WaveEquationSolver1D);

	protected:
		void AdvanceImpl(cl::ColumnWiseMatrix<memorySpace, mathDomain>& solution,
						 const std::shared_ptr<cl::Tensor<memorySpace, mathDomain>>& timeDiscretizers,
						 const SolverType solverType,
						 const unsigned nSteps = 1);

		void MakeTimeDiscretizer(const std::shared_ptr<cl::Tensor<memorySpace, mathDomain>>& timeDiscretizers, const SolverType solverType);

		void Setup(const unsigned solverSteps);
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

#include <WaveEquationSolver1D.tpp>