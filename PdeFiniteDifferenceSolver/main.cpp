#include <Utils/PdeSetup.h>

template<typename solverType, MathDomain md>
void run(solverType& solver, const clp::CommandLineArgumentParser& ap, bool debug)
{
	std::chrono::high_resolution_clock::time_point  start, end;

	using vType = cl::Vector<MemorySpace::Device, md>;
	using sType = typename vType::stdType;
	std::vector<sType> solutionMatrix;

	// steps to advance before outputing the solution
	auto n = ap.GetArgumentValue<unsigned>("-n");

	// total number of steps
	auto N = ap.GetArgumentValue<unsigned>("-N");

	auto outputFileString = ap.GetArgumentValue<std::string>("-of", "sol.cl");

	unsigned nSolutions = 0;
	DEBUG_PRINT_START(Solving ...)

	for (unsigned m = 0; m < N; ++m)
	{
		solver.Advance(n);

		const auto solution = solver.solution->columns[0]->Get();
		solutionMatrix.insert(solutionMatrix.end(), solution.begin(), solution.end());
		++nSolutions;
	}

	DEBUG_PRINT_END

	DEBUG_PRINT_START(Saving to file ...)
	std::ofstream outputFile(outputFileString);
	cl::MatrixToBinaryFile<sType>(solutionMatrix, nSolutions, solver.solution->columns[0]->size(), outputFileString, false);
	DEBUG_PRINT_END
}

int main(int argc, char** argv)
{
	clp::CommandLineArgumentParser ap(argc, argv);

	auto mathDomain = ep::ParseMathDomain(ap.GetArgumentValue<std::string>("-md", "Float"));
	auto pdeType = ap.GetArgumentValue<std::string>("-pde", "AdvectionDiffusion");
	auto dimensionality = ap.GetArgumentValue<int>("-dim", 1);
	auto debug = ap.GetFlag("-dbg");

	if (dimensionality == 1)
	{
		switch (mathDomain)
		{
			case MathDomain::Float:
				if (pdeType == "AdvectionDiffusion")
				{
					auto solver = setup1D<pde::AdvectionDiffusionSolver1D<MemorySpace::Device, MathDomain::Float>, MathDomain::Float>(ap, debug);
					run<decltype(*solver), MathDomain::Float>(*solver, ap, debug);
				}
				else if (pdeType == "WaveEquation")
				{
					auto solver = setup1D<pde::WaveEquationSolver1D<MemorySpace::Device, MathDomain::Float>, MathDomain::Float>(ap, debug);
					run<decltype(*solver), MathDomain::Float>(*solver, ap, debug);
				}
				else
					throw NotImplementedException();
				break;
			case MathDomain::Double:
				if (pdeType == "AdvectionDiffusion")
				{
					auto solver = setup1D<pde::AdvectionDiffusionSolver1D<MemorySpace::Device, MathDomain::Double>, MathDomain::Double>(ap, debug);
					run<decltype(*solver), MathDomain::Double>(*solver, ap, debug);
				}
				else if (pdeType == "WaveEquation")
				{
					auto solver = setup1D<pde::WaveEquationSolver1D<MemorySpace::Device, MathDomain::Double>, MathDomain::Double>(ap, debug);
					run<decltype(*solver), MathDomain::Double>(*solver, ap, debug);
				}
				else
					throw NotImplementedException();
				break;
			default:
				throw NotImplementedException();
		}
	}
	else if (dimensionality == 2)
	{
		switch (mathDomain)
		{
			case MathDomain::Float:
				if (pdeType == "AdvectionDiffusion")
				{
					auto solver = setup2D<pde::AdvectionDiffusionSolver2D<MemorySpace::Device, MathDomain::Float>, MathDomain::Float>(ap, debug);
					run<decltype(*solver), MathDomain::Float>(*solver, ap, debug);
				}
				else if (pdeType == "WaveEquation")
				{
					auto solver = setup2D<pde::WaveEquationSolver2D<MemorySpace::Device, MathDomain::Float>, MathDomain::Float>(ap, debug);
					run<decltype(*solver), MathDomain::Float>(*solver, ap, debug);
				}
				else
					throw NotImplementedException();
				break;
			case MathDomain::Double:
				if (pdeType == "AdvectionDiffusion")
				{
					auto solver = setup2D<pde::AdvectionDiffusionSolver2D<MemorySpace::Device, MathDomain::Double>, MathDomain::Double>(ap, debug);
					run<decltype(*solver), MathDomain::Float>(*solver, ap, debug);
				}
				else if (pdeType == "WaveEquation")
				{
					auto solver = setup2D<pde::WaveEquationSolver2D<MemorySpace::Device, MathDomain::Double>, MathDomain::Double>(ap, debug);
					run<decltype(*solver), MathDomain::Double>(*solver, ap, debug);
				}
				else
					throw NotImplementedException();
				break;
			default:
				throw NotImplementedException();
		}
	}
	else
	{
		throw NotImplementedException();
	}

	return 0;
}