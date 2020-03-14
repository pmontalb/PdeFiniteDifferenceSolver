#pragma once

namespace ep
{
#define PARSE(E, X)\
    if (!strcmp(text.c_str(), #X))\
        return E::X;

    static inline SolverType ParseSolverType(const std::string &text) noexcept {
#define PARSE_WORKER(X) PARSE(SolverType, X)

        PARSE_WORKER(ExplicitEuler)
        PARSE_WORKER(ImplicitEuler)
        PARSE_WORKER(CrankNicolson)
        PARSE_WORKER(RungeKuttaRalston)
        PARSE_WORKER(RungeKutta3)
        PARSE_WORKER(RungeKutta4)
        PARSE_WORKER(RungeKuttaThreeEight)
        PARSE_WORKER(RungeKuttaGaussLegendre4)
        PARSE_WORKER(RichardsonExtrapolation2)
        PARSE_WORKER(RichardsonExtrapolation3)
        PARSE_WORKER(AdamsBashforth2)
        PARSE_WORKER(AdamsMouldon2)

#undef PARSE_WORKER

        return SolverType::Null;
    }

    static inline SpaceDiscretizerType ParseSpaceDiscretizer(const std::string &text) noexcept {
#define PARSE_WORKER(X) PARSE(SpaceDiscretizerType, X)

        PARSE_WORKER(Centered)
        PARSE_WORKER(Upwind)
        PARSE_WORKER(LaxWendroff)

#undef PARSE_WORKER
        return SpaceDiscretizerType::Null;
    }

    static inline BoundaryConditionType ParseBoundaryConditionType(const std::string &text) noexcept {
#define PARSE_WORKER(X) PARSE(BoundaryConditionType, X);

        PARSE_WORKER(Dirichlet)
        PARSE_WORKER(Neumann)
        PARSE_WORKER(Periodic)

#undef PARSE_WORKER
        return BoundaryConditionType::Null;
    }

    static inline MathDomain ParseMathDomain(const std::string &text) noexcept {
#define PARSE_WORKER(X) PARSE(MathDomain, X)

        PARSE_WORKER(Double)
        PARSE_WORKER(Float)

#undef PARSE_WORKER
        return MathDomain::Null;
    }

#undef PARSE

}
