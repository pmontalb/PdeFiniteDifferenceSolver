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

		switch (inputData.solverType)
		{
			case SolverType::ExplicitEuler:
				sparseExplicitTimeDiscretizer = std::make_unique<cl::CompressedSparseRowMatrix<ms, md>>(*this->timeDiscretizers->matrices[0]);
				timeDiscretizers.reset();
			default:
				break;
		}
	}

	template<class pdeImpl, class pdeInputType, MemorySpace ms, MathDomain md>
	void FiniteDifferenceSolver<pdeImpl, pdeInputType, ms, md>::Advance(const unsigned nSteps)
	{
		if (!hasPrecomputed)
		{
			hasPrecomputed = true;
			Precompute();
		}

		if (timeDiscretizers)
		{
			assert(sparseExplicitTimeDiscretizer == nullptr);
			static_cast<pdeImpl *>(this)->AdvanceImpl(*solution, *timeDiscretizers, inputData.solverType, nSteps);
		}
		else
		{
			assert(timeDiscretizers == nullptr);
			static_cast<pdeImpl *>(this)->AdvanceImpl(*solution, *sparseExplicitTimeDiscretizer, inputData.solverType, nSteps);
		}
	}
}
