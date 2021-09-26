
#include <cmath>
#include <gtest/gtest.h>

#include <ColumnWiseMatrix.h>
#include <Vector.h>

#include <IterableEnum.h>
#include <WaveEquationSolver1D.h>

namespace pdet
{
	class WaveEquation1DTests: public ::testing::Test
	{
	};

	TEST_F(WaveEquation1DTests, ConstantSolutionNoTransport)
	{
		cl::vec initialCondition(10, 1.0f);
		cl::vec grid = cl::vec::LinSpace(0.0f, 1.0f, initialCondition.size());
		float dt = 1e-4f;
		float velocity = 0.0f;
		float diffusion = 0.0f;

		for (const SolverType solverType : enums::IterableEnum<SolverType>())
		{
			if (solverType != SolverType::ExplicitEuler && solverType != SolverType::ImplicitEuler)
				continue;

			pde::GpuSinglePdeInputData1D data(initialCondition, grid, velocity, diffusion, dt, solverType, SpaceDiscretizerType::Centered);
			pde::wave1D solver(std::move(data));

			const auto _initialCondition = solver.inputData.initialCondition.Get();
			for (unsigned n = 1; n < 10; ++n)
			{
				solver.Advance(n);
				const auto solution = solver.solution->columns[0]->Get();

				for (size_t i = 0; i < solution.size(); ++i)
					ASSERT_TRUE(std::fabs(solution[i] - _initialCondition[i]) <= 1e-15f);
			}
		}
	}

	TEST_F(WaveEquation1DTests, ConstantSolution)
	{
		cl::vec initialCondition(10, 1.0f);
		cl::vec grid = cl::vec::LinSpace(0.0f, 1.0f, initialCondition.size());
		float dt = 1e-4f;
		float velocity = 1.0f;
		float diffusion = 2.0f;

		for (const SolverType solverType : enums::IterableEnum<SolverType>())
		{
			if (solverType != SolverType::ExplicitEuler && solverType != SolverType::ImplicitEuler)
				continue;

			pde::GpuSinglePdeInputData1D data(initialCondition, grid, velocity, diffusion, dt, solverType, SpaceDiscretizerType::Centered);
			pde::wave1D solver(std::move(data));

			const auto _initialCondition = solver.inputData.initialCondition.Get();
			for (unsigned n = 1; n < 10; ++n)
			{
				solver.Advance(n);
				const auto solution = solver.solution->columns[0]->Get();

				for (size_t i = 0; i < solution.size(); ++i)
					ASSERT_TRUE(std::fabs(solution[i] - _initialCondition[i]) <= 1e-15f);
			}
		}
	}

	TEST_F(WaveEquation1DTests, LinearSolution)
	{
		cl::dvec grid = cl::dvec::LinSpace(0.0, 1.0, 10);
		auto _grid = grid.Get();

		const auto f = [](const double x) { return 2.0 * x + 1; };

		std::vector<double> _initialCondition(10);
		for (unsigned i = 0; i < _initialCondition.size(); ++i)
			_initialCondition[i] = f(_grid[i]);

		cl::dvec initialCondition(_initialCondition);

		unsigned steps = 1000;
		double dt = 1e-4;
		double velocity = .5;
		double diffusion = 0.0;

		double finalTime = steps * dt;
		std::vector<double> _exactSolution(10);
		for (unsigned i = 0; i < _initialCondition.size(); ++i)
			_exactSolution[i] = .5 * (f(_grid[i] - velocity * finalTime) + f(_grid[i] + velocity * finalTime));

		// need to setup the correct boundary condition with the slope of the line
		BoundaryCondition leftBoundaryCondition(BoundaryConditionType::Neumann, 2.0);
		BoundaryCondition rightBoundaryCondition(BoundaryConditionType::Neumann, -2.0);
		BoundaryCondition1D boundaryConditions(leftBoundaryCondition, rightBoundaryCondition);
		for (const SolverType solverType : enums::IterableEnum<SolverType>())
		{
			if (solverType != SolverType::ExplicitEuler && solverType != SolverType::ImplicitEuler)
				continue;

			pde::GpuDoublePdeInputData1D data(initialCondition, grid, velocity, diffusion, dt, solverType, SpaceDiscretizerType::Centered, boundaryConditions);
			pde::dwave1D solver(std::move(data));

			solver.Advance(steps);
			const auto solution = solver.solution->columns[0]->Get();

			for (size_t i = 0; i < solution.size(); ++i)
				ASSERT_TRUE(fabs(solution[i] - _exactSolution[i]) <= 5e-6) << "err=" << fabs(solution[i] - _exactSolution[i]);
		}
	}
}	 // namespace pdet
