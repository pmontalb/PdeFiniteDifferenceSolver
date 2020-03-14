#pragma once

#include <FiniteDifferenceSolver.h>

namespace pde
{
	template<class pdeImpl, class pdeInputType, MemorySpace ms, MathDomain md>
	FiniteDifferenceSolver<pdeImpl, pdeInputType, ms, md>::FiniteDifferenceSolver(pdeInputType&& inputData_)
		: inputData(std::move(inputData_))
	{
	}

	template<class pdeImpl, class pdeInputType, MemorySpace ms, MathDomain md>
	void FiniteDifferenceSolver<pdeImpl, pdeInputType, ms, md>::Precompute()
	{
		static_cast<pdeImpl*>(this)->Setup(getNumberOfSteps(inputData.solverType));
		static_cast<pdeImpl*>(this)->MakeTimeDiscretizer(*this->timeDiscretizers, inputData.solverType);
	}

	template<class pdeImpl, class pdeInputType, MemorySpace ms, MathDomain md>
	void FiniteDifferenceSolver<pdeImpl, pdeInputType, ms, md>::Advance(const unsigned nSteps)
	{
		if (!hasPrecomputed)
		{
			hasPrecomputed = true;
			Precompute();
		}
		static_cast<pdeImpl*>(this)->AdvanceImpl(*solution, *timeDiscretizers, inputData.solverType, nSteps);
	}
}
