
#include <gtest/gtest.h>

#include <Vector.h>
#include <ColumnWiseMatrix.h>

#include <WaveEquationSolver1D.h>
#include <IterableEnum.h>

namespace pdet
{
	class WaveEquationTests : public ::testing::Test
	{

	};

	TEST_F(WaveEquationTests, ConstantSolutionNoTransport)
	{
		cl::vec initialCondition(10, 1.0f);
		cl::vec grid = cl::LinSpace(0.0f, 1.0f, initialCondition.size());
		double dt = 1e-4;
		float velocity = 0.0f;
		float diffusion = 0.0f;

		for (const SolverType solverType : enums::IterableEnum<SolverType>())
		{
			if (solverType != SolverType::ExplicitEuler && solverType != SolverType::ImplicitEuler)
				continue;

			pde::GpuSinglePdeInputData data(initialCondition, grid, velocity, diffusion, dt, solverType, SpaceDiscretizerType::Centered);
			pde::wave1D solver(data);

			const auto _initialCondition = solver.inputData.initialCondition.Get();
			for (unsigned n = 1; n < 10; ++n)
			{
				solver.Advance(n);
				const auto solution = solver.solution->columns[0]->Get();

				for (size_t i = 0; i < solution.size(); ++i)
					ASSERT_TRUE(fabs(solution[i] - _initialCondition[i]) <= 1e-15);
			}
		}
	}

	TEST_F(WaveEquationTests, ConstantSolution)
	{
		cl::vec initialCondition(10, 1.0f);
		cl::vec grid = cl::LinSpace(0.0f, 1.0f, initialCondition.size());
		double dt = 1e-4;
		float velocity = 1.0f;
		float diffusion = 2.0f;

		for (const SolverType solverType : enums::IterableEnum<SolverType>())
		{
			if (solverType != SolverType::ExplicitEuler && solverType != SolverType::ImplicitEuler)
				continue;

			pde::GpuSinglePdeInputData data(initialCondition, grid, velocity, diffusion, dt, solverType, SpaceDiscretizerType::Centered);
			pde::wave1D solver(data);

			const auto _initialCondition = solver.inputData.initialCondition.Get();
			for (unsigned n = 1; n < 10; ++n)
			{
				solver.Advance(n);
				const auto solution = solver.solution->columns[0]->Get();

				for (size_t i = 0; i < solution.size(); ++i)
					ASSERT_TRUE(fabs(solution[i] - _initialCondition[i]) <= 1e-15);
			}
		}
	}

	TEST_F(WaveEquationTests, LinearSolution)
	{
		cl::dvec grid = cl::LinSpace<MemorySpace::Device, MathDomain::Double>(0.0, 1.0, 10);
		auto _grid = grid.Get();

		const auto f = [](const double x)
		{
			return 2.0 * x + 1;
		};

		std::vector<double> _initialCondition(10);
		for (unsigned i = 0; i < _initialCondition.size(); ++i)
			_initialCondition[i] = f(_grid[i]);

		cl::dvec initialCondition(_initialCondition);

		unsigned steps = 1000;
		double dt = 1e-4;
		float velocity = .5f;
		float diffusion = 0.0f;

		float finalTime = steps * dt;
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

			pde::GpuDoublePdeInputData data(initialCondition, grid, velocity, diffusion, dt, solverType, SpaceDiscretizerType::Centered, boundaryConditions);
			pde::dwave1D solver(data);

			solver.Advance(steps);
			const auto solution = solver.solution->columns[0]->Get();

			for (size_t i = 0; i < solution.size(); ++i)
				ASSERT_TRUE(fabs(solution[i] - _exactSolution[i]) <= 1e-8);

		}
	}
}