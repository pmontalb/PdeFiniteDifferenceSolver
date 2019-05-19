#pragma once

#include <FiniteDifferenceSolver.h>

namespace pde
{
	template<class pdeImpl, class pdeInputType, MemorySpace ms, MathDomain md>
	FiniteDifferenceSolver<pdeImpl, pdeInputType, ms, md>::FiniteDifferenceSolver(const pdeInputType& inputData)
		: inputData(inputData)
	{
		static_cast<pdeImpl*>(this)->Setup(getNumberOfSteps(inputData.solverType));
		static_cast<pdeImpl*>(this)->MakeTimeDiscretizer(this->timeDiscretizers, inputData.solverType);
	}

	template<class pdeImpl, class pdeInputType, MemorySpace ms, MathDomain md>
	void FiniteDifferenceSolver<pdeImpl, pdeInputType, ms, md>::Advance(const unsigned nSteps)
	{
		static_cast<pdeImpl*>(this)->AdvanceImpl(*solution, timeDiscretizers, inputData.solverType, nSteps);
	}

	template<class pdeImpl, class pdeInputType, MemorySpace ms, MathDomain md>
	const cl::Tensor<ms, md>* FiniteDifferenceSolver<pdeImpl, pdeInputType, ms, md>::GetTimeDiscretizer() const noexcept
	{
		return timeDiscretizers ? timeDiscretizers.get() : nullptr;
	}
}