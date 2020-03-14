#pragma once

namespace clp
{
    class IllegalArgumentException: public Exception
    {
    public:
        IllegalArgumentException(const std::string &message = "")
                : Exception("IllegalArgumentException: " + message)
        {
        }
    };

    class ArgumentNotFoundException: public Exception
    {
    public:
        ArgumentNotFoundException(const std::string &message = "")
                : Exception("ArgumentNotFoundException: " + message)
        {
        }
    };

    class CommandLineArgumentParser
    {
    public:
        CommandLineArgumentParser(int argc, char **argv)
                : args(argv, argv + argc)
        {
        }

        template<typename T>
        T GetArgumentValue(const std::string &option) const;

        template<typename T>
        T GetArgumentValue(const std::string &option, const T &defaultValue) const noexcept
        {
            T ret;
            try
            {
                ret = GetArgumentValue<T>(option);
            }
            catch (ArgumentNotFoundException &)
            {
                ret = defaultValue;
            }

            return ret;
        }

        bool GetFlag(const std::string &option) const
        {
            return std::find(args.begin(), args.end(), option) != args.end();
        }

    private:
        std::vector <std::string> args;
    };

    template<>
    std::string CommandLineArgumentParser::GetArgumentValue<std::string>(const std::string &option) const
    {
        auto itr = std::find(args.begin(), args.end(), option);
        if (itr != args.end())
        {
            if (++itr == args.end())
                throw IllegalArgumentException(option);
            return *itr;
        }

        throw ArgumentNotFoundException(option);
    }

    template<>
    int CommandLineArgumentParser::GetArgumentValue<int>(const std::string &option) const
    {
        return std::atoi(GetArgumentValue<std::string>(option).c_str());
    }

    template<>
    short CommandLineArgumentParser::GetArgumentValue<short>(const std::string &option) const
    {
        return static_cast<short>(std::atoi(GetArgumentValue<std::string>(option).c_str()));
    }

    template<>
    unsigned CommandLineArgumentParser::GetArgumentValue<unsigned>(const std::string &option) const
    {
        return static_cast<unsigned>(std::atoi(GetArgumentValue<std::string>(option).c_str()));
    }

    template<>
    long CommandLineArgumentParser::GetArgumentValue<long>(const std::string &option) const
    {
        return static_cast<long>(std::atoi(GetArgumentValue<std::string>(option).c_str()));
    }

    template<>
    long long CommandLineArgumentParser::GetArgumentValue<long long>(const std::string &option) const
    {
        return static_cast<long long>(std::atoi(GetArgumentValue<std::string>(option).c_str()));
    }

    template<>
    double CommandLineArgumentParser::GetArgumentValue<double>(const std::string &option) const
    {
        return std::atof(GetArgumentValue<std::string>(option).c_str());
    }

    template<>
    float CommandLineArgumentParser::GetArgumentValue<float>(const std::string &option) const
    {
        return static_cast<float>(std::atof(GetArgumentValue<std::string>(option).c_str()));
    }
}
