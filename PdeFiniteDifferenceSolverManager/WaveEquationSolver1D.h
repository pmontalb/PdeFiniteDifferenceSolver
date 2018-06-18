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
		friend class FiniteDifferenceSolver1D<WaveEquationSolver1D<memorySpace, mathDomain>, memorySpace, mathDomain>;
		using FiniteDifferenceSolver1D::FiniteDifferenceSolver1D;

		MAKE_DEFAULT_CONSTRUCTORS(WaveEquationSolver1DS);

	protected:
		void MakeTimeDiscretizerWorker(const std::shared_ptr<cl::Tensor<memorySpace, mathDomain>>& timeDiscretizers,
									   const cl::ColumnWiseMatrix<memorySpace, mathDomain>& spaceDiscretizer,
									   const FiniteDifferenceInput1D& input)
		{
			pde::detail::MakeTimeDiscretizerWaveEquation(timeDiscretizers->GetCube(), spaceDiscretizer.GetTile(), input.solverType, input.dt);
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

