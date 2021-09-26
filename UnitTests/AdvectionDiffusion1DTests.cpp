
#include <gtest/gtest.h>

#include <ColumnWiseMatrix.h>
#include <Vector.h>

#include <AdvectionDiffusionSolver1D.h>
#include <IterableEnum.h>

namespace pdet
{
	class AdvectionDiffusion1DTests: public ::testing::Test
	{
	};

	TEST_F(AdvectionDiffusion1DTests, ConstantSolutionNoTransportNoDiffusion)
	{
		cl::vec initialCondition(10, 1.0f);
		cl::vec grid = cl::vec::LinSpace(0.0f, 1.0f, initialCondition.size());
		float dt = 1e-4f;
		float velocity = 0.0f;
		float diffusion = 0.0f;

		for (const SolverType solverType : enums::IterableEnum<SolverType>())
		{
			if (solverType != SolverType::AdamsBashforth2)
				continue;
			pde::GpuSinglePdeInputData1D data(initialCondition, grid, velocity, diffusion, dt, solverType, SpaceDiscretizerType::Centered);
			pde::ad1D solver(std::move(data));

			const auto _initialCondition = solver.inputData.initialCondition.Get();
			for (unsigned n = 1; n < 10; ++n)
			{
				solver.Advance(n);
				const auto solution = solver.solution->columns[0]->Get();

				for (size_t i = 0; i < solution.size(); ++i)
					EXPECT_TRUE(std::fabs(solution[i] - _initialCondition[i]) <= 1e-7f) << static_cast<int>(solverType);
			}
		}
	}

	TEST_F(AdvectionDiffusion1DTests, ConstantSolutionNoDiffusion)
	{
		cl::vec initialCondition(10, 1.0f);
		cl::vec grid = cl::vec::LinSpace(0.0f, 1.0f, initialCondition.size());
		float dt = 1e-4f;
		float velocity = 1.0f;
		float diffusion = 0.0f;

		for (const SolverType solverType : enums::IterableEnum<SolverType>())
		{
			pde::GpuSinglePdeInputData1D data(initialCondition, grid, velocity, diffusion, dt, solverType, SpaceDiscretizerType::Centered);
			pde::ad1D solver(std::move(data));

			const auto _initialCondition = solver.inputData.initialCondition.Get();
			for (unsigned n = 0; n < 10; ++n)
			{
				solver.Advance(n);
				const auto solution = solver.solution->columns[0]->Get();

				for (size_t i = 0; i < solution.size(); ++i)
				{
					if (solverType != SolverType::RichardsonExtrapolation2 && solverType != SolverType::RichardsonExtrapolation3)
						ASSERT_TRUE(std::fabs(solution[i] - _initialCondition[i]) <= 2.8e-6f);
					else
						ASSERT_TRUE(std::fabs(solution[i] - _initialCondition[i]) <= 3e-5f);
				}
			}
		}
	}

	TEST_F(AdvectionDiffusion1DTests, ConstantSolution)
	{
		cl::vec initialCondition(10, 1.0f);
		cl::vec grid = cl::vec::LinSpace(0.0f, 1.0f, initialCondition.size());
		float dt = 1e-4f;
		float velocity = 1.0f;
		float diffusion = 2.0f;

		for (const SolverType solverType : enums::IterableEnum<SolverType>())
		{
			pde::GpuSinglePdeInputData1D data(initialCondition, grid, velocity, diffusion, dt, solverType, SpaceDiscretizerType::Centered);
			pde::ad1D solver(std::move(data));

			const auto _initialCondition = solver.inputData.initialCondition.Get();
			for (unsigned n = 1; n < 10; ++n)
			{
				solver.Advance(n);
				const auto solution = solver.solution->columns[0]->Get();

				for (size_t i = 0; i < solution.size(); ++i)
				{
					if (solverType != SolverType::RichardsonExtrapolation2 && solverType != SolverType::RichardsonExtrapolation3)
						EXPECT_TRUE(std::fabs(solution[i] - _initialCondition[i]) <= 5.5e-6f) << static_cast<int>(solverType) << "|" << std::fabs(solution[i] - _initialCondition[i]);
					else
						ASSERT_TRUE(std::fabs(solution[i] - _initialCondition[i]) <= 3e-5f);
				}
			}
		}
	}

	TEST_F(AdvectionDiffusion1DTests, LinearSolutionNoTransport)
	{
		cl::vec initialCondition = cl::vec::LinSpace(0.0f, 10.0f, 10);
		cl::vec grid = cl::vec::LinSpace(0.0f, 1.0f, initialCondition.size());
		float dt = 1e-4f;
		float velocity = 0.0f;
		float diffusion = 2.0f;

		for (const SolverType solverType : enums::IterableEnum<SolverType>())
		{
			// need to setup the correct boundary condition with the slope of the line
			BoundaryCondition leftBoundaryCondition(BoundaryConditionType::Neumann, 10.0);
			BoundaryCondition rightBoundaryCondition(BoundaryConditionType::Neumann, -10.0);
			BoundaryCondition1D boundaryConditions(leftBoundaryCondition, rightBoundaryCondition);

			pde::GpuSinglePdeInputData1D data(initialCondition, grid, velocity, diffusion, dt, solverType, SpaceDiscretizerType::Centered, boundaryConditions);
			pde::ad1D solver(std::move(data));

			const auto _initialCondition = solver.inputData.initialCondition.Get();
			for (unsigned n = 1; n < 10; ++n)
			{
				solver.Advance(n);
				const auto solution = solver.solution->columns[0]->Get();

				for (size_t i = 0; i < solution.size(); ++i)
				{
					if (solverType != SolverType::RichardsonExtrapolation2 && solverType != SolverType::RichardsonExtrapolation3)
						ASSERT_TRUE(std::fabs(solution[i] - _initialCondition[i]) <= 2.7e-5f);
					else
						ASSERT_TRUE(std::fabs(solution[i] - _initialCondition[i]) <= 5e-4f);
				}
			}
		}
	}

	TEST_F(AdvectionDiffusion1DTests, SineSolutionNoDiffusion)
	{
		cl::vec grid = cl::vec::LinSpace(0.0f, 1.0f, 10);
		auto _grid = grid.Get();

		std::vector<float> _initialCondition(10);
		for (unsigned i = 0; i < _initialCondition.size(); ++i)
			_initialCondition[i] = std::sin(_grid[i]);

		cl::vec initialCondition(_initialCondition);

		unsigned steps = 10;
		auto dt = 1e-4f;
		auto velocity = .05f;
		auto diffusion = 0.0f;

		auto finalTime = static_cast<float>(steps) * dt;
		std::vector<float> _exactSolution(10);
		for (unsigned i = 0; i < _initialCondition.size(); ++i)
			_exactSolution[i] = std::sin(_grid[i] - velocity * finalTime);

		for (const SolverType solverType : enums::IterableEnum<SolverType>())
		{
			pde::GpuSinglePdeInputData1D data(initialCondition, grid, velocity, diffusion, dt, solverType, SpaceDiscretizerType::Centered);
			pde::ad1D solver(std::move(data));

			solver.Advance(steps);
			const auto solution = solver.solution->columns[0]->Get();

			for (size_t i = 1; i < solution.size() - 1; ++i)	// excluding BC
				ASSERT_TRUE(std::fabs(solution[i] - _exactSolution[i]) <= 1.1e-4f);
		}
	}
}	 // namespace pdet
