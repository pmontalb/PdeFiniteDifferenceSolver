#pragma once

#include <memory>
#include <Vector.h>
#include <IBuffer.h>
#include <Types.h>

#include <PdeInputData.h>
#include <FiniteDifferenceManager.h>

#define MAKE_DEFAULT_CONSTRUCTORS(CLASS)\
	virtual ~CLASS() noexcept = default;\
	CLASS(const CLASS& rhs) noexcept = default;\
	CLASS(CLASS&& rhs) noexcept = default;\
	CLASS& operator=(const CLASS& rhs) noexcept = default;\
	CLASS& operator=(CLASS&& rhs) noexcept = default;\


namespace pde
{
	/**
	*	CRTP implementation
	*	Instead of using traits, I decided to pass another template parameter - pdeInputType - for a less verbose code
	*/
	template<class pdeImpl, class pdeInputType, MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class FiniteDifferenceSolver
	{
	public:
		FiniteDifferenceSolver(const pdeInputType& inputData)
			: inputData(inputData)
		{
			static_cast<pdeImpl*>(this)->Setup(getNumberOfSteps(inputData.solverType));

			static_cast<pdeImpl*>(this)->MakeTimeDiscretizer();
		}

		MAKE_DEFAULT_CONSTRUCTORS(FiniteDifferenceSolver);

		void Advance(const unsigned nSteps = 1)
		{
			static_cast<pdeImpl*>(this)->AdvanceImpl(nSteps);
		}

		const cl::Tensor<memorySpace, mathDomain>* const GetTimeDiscretizer() const noexcept { return timeDiscretizers ? timeDiscretizers.get() : nullptr; }

		std::shared_ptr<cl::ColumnWiseMatrix<memorySpace, mathDomain>> solution;
	protected:
		const pdeInputType& inputData;
		std::shared_ptr<cl::Tensor<memorySpace, mathDomain>> timeDiscretizers;
	};

	template<MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class FiniteDifferenceSolver1D : public FiniteDifferenceSolver<FiniteDifferenceSolver1D<memorySpace, mathDomain>, PdeInputData1D<memorySpace, mathDomain>, memorySpace, mathDomain>
	{
	public:
		friend class FiniteDifferenceSolver<FiniteDifferenceSolver1D<memorySpace, mathDomain>, PdeInputData1D<memorySpace, mathDomain>, memorySpace, mathDomain>;
		using FiniteDifferenceSolver::FiniteDifferenceSolver;

		MAKE_DEFAULT_CONSTRUCTORS(FiniteDifferenceSolver1D);

	protected:
		void MakeTimeDiscretizer()
		{
			// needs to set 0 everywhere in the space discretizer
			cl::ColumnWiseMatrix<memorySpace, mathDomain> spaceDiscretizer(solution->nRows(), solution->nRows(), 0.0);

			FiniteDifferenceInput1D _input(inputData.dt,
										   inputData.spaceGrid.GetBuffer(),
										   inputData.velocity.GetBuffer(),
										   inputData.diffusion.GetBuffer(),
										   inputData.solverType, 
										   inputData.boundaryConditions);
			pde::detail::MakeSpaceDiscretizer1D(spaceDiscretizer.GetTile(), _input);
			pde::detail::MakeTimeDiscretizer1D(timeDiscretizers->GetCube(), spaceDiscretizer.GetTile(), _input);
		}

		void AdvanceImpl(const unsigned nSteps = 1)
		{
			FiniteDifferenceInput1D _input(inputData.dt,
										   inputData.spaceGrid.GetBuffer(),
										   inputData.velocity.GetBuffer(),
										   inputData.diffusion.GetBuffer(),
										   inputData.solverType,
										   inputData.boundaryConditions);
			pde::detail::Iterate1D(solution->GetTile(), timeDiscretizers->GetCube(), _input, nSteps);
		}

		void Setup(const unsigned solverSteps)
		{
			this->solution = std::make_shared<cl::ColumnWiseMatrix<memorySpace, mathDomain>>(this->inputData.initialCondition.nRows(), solverSteps);
			this->solution->Set(*this->inputData.initialCondition.matrices[0]->columns[0], 0);

			// needs to set 0 everywhere in the time discretizer
			this->timeDiscretizers = std::make_shared<cl::Tensor<memorySpace, mathDomain>>(this->inputData.initialCondition.nRows(), this->inputData.initialCondition.nRows(), solverSteps, 0.0);
		}
	};

	template<MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class FiniteDifferenceSolver2D : public FiniteDifferenceSolver<FiniteDifferenceSolver2D<memorySpace, mathDomain>, PdeInputData2D<memorySpace, mathDomain>, memorySpace, mathDomain>
	{
	public:
		friend class FiniteDifferenceSolver<PdeInputData2D<memorySpace, mathDomain>, PdeInputData2D<memorySpace, mathDomain>, memorySpace, mathDomain>;
		using FiniteDifferenceSolver::FiniteDifferenceSolver;

		MAKE_DEFAULT_CONSTRUCTORS(FiniteDifferenceSolver2D);

	protected:
		void MakeTimeDiscretizer()
		{
		}

		void AdvanceImpl(const unsigned nSteps = 1)
		{
		}

		void Setup(const unsigned solverSteps)
		{
			const unsigned dimension = this->inputData.initialCondition.nRows(), this->inputData.initialCondition.nCols();

			this->solution = std::make_shared<cl::ColumnWiseMatrix<memorySpace, mathDomain>>(dimension * dimension, solverSteps);
			this->solution->Set(this->inputData.initialCondition.matrices[0], 0);

			// needs to set 0 everywhere in the time discretizer
			this->timeDiscretizers = std::make_shared<cl::Tensor<memorySpace, mathDomain>>(dimension, dimension, solverSteps);
		}
	};

    #pragma region Type aliases

#define MAKE_TYPE_ALIAS(DIM)\
	typedef FiniteDifferenceSolver##DIM<MemorySpace::Device, MathDomain::Float> GpuSinglePdeSolver##DIM;\
	typedef GpuSinglePdeSolver##DIM GpuFloatSolver##DIM;\
	typedef FiniteDifferenceSolver##DIM<MemorySpace::Device, MathDomain::Double> GpuDoublePdeSolver##DIM;\
	typedef FiniteDifferenceSolver##DIM<MemorySpace::Host, MathDomain::Float> CpuSinglePdeSolver##DIM;\
	typedef CpuSinglePdeSolver##DIM CpuFloatSolver##DIM;\
	typedef FiniteDifferenceSolver##DIM<MemorySpace::Host, MathDomain::Double> CpuDoubleSolver##DIM;\
	typedef GpuSinglePdeSolver##DIM sol##DIM;\
	typedef GpuDoublePdeSolver##DIM dsol##DIM;

	MAKE_TYPE_ALIAS(1D);
	MAKE_TYPE_ALIAS(2D);

#undef MAKE_TYPE_ALIAS

    #pragma endregion
}

#undef MAKE_DEFAULT_CONSTRUCTORS