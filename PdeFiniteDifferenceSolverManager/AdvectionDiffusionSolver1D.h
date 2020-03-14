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
	class AdvectionDiffusionSolver1D : public FiniteDifferenceSolver1D<AdvectionDiffusionSolver1D<memorySpace, mathDomain>, memorySpace, mathDomain>
	{
	public:
		// befriend the grandparent CRTP class
		friend class FiniteDifferenceSolver<AdvectionDiffusionSolver1D<memorySpace, mathDomain>, PdeInputData1D<memorySpace, mathDomain>, memorySpace, mathDomain>;
		// befriend the mother CRTP class
		friend class FiniteDifferenceSolver1D<AdvectionDiffusionSolver1D<memorySpace, mathDomain>, memorySpace, mathDomain>;
		
		using FiniteDifferenceSolver1D<AdvectionDiffusionSolver1D<memorySpace, mathDomain>, memorySpace, mathDomain>::FiniteDifferenceSolver1D;

		MAKE_DEFAULT_CONSTRUCTORS(AdvectionDiffusionSolver1D)

	protected:
		void MakeTimeDiscretizer(cl::Tensor<memorySpace, mathDomain>& timeDiscretizers_, const SolverType solverType);
	};

#pragma region Type aliases

	typedef AdvectionDiffusionSolver1D<MemorySpace::Device, MathDomain::Float> GpuSingleAdvectionDiffusionSolver1D;
	typedef GpuSingleAdvectionDiffusionSolver1D GpuFloatAdvectionDiffusionSolver1D;
	typedef AdvectionDiffusionSolver1D<MemorySpace::Device, MathDomain::Double> GpuDoubleAdvectionDiffusionSolver1D;
	typedef AdvectionDiffusionSolver1D<MemorySpace::Host, MathDomain::Float> CpuSingleAdvectionDiffusionSolver1D;
	typedef CpuSingleAdvectionDiffusionSolver1D CpuFloatAdvectionDiffusionSolver1D;
	typedef AdvectionDiffusionSolver1D<MemorySpace::Host, MathDomain::Double> CpuDoubleAdvectionDiffusionSolver1D;
	typedef GpuSingleAdvectionDiffusionSolver1D ad1D;
	typedef GpuDoubleAdvectionDiffusionSolver1D dad1D;

#pragma endregion
}

#undef MAKE_DEFAULT_CONSTRUCTORS

#include <AdvectionDiffusionSolver1D.tpp>

