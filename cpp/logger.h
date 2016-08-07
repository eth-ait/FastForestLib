//
//  logger.h
//  DistRandomForest
//
//  Created by Benjamin Hepp on 09/10/15.
//
//

#pragma once

#include <iostream>
#include <fstream>

namespace ait
{

class Logger
{
    std::string prefix_;
    
public:
    class LogStream
    {
        std::ostream* sout_;
        bool new_line_;
        bool flush_;
        bool closed_;
        bool allocated_;
        
    public:
        explicit LogStream()
        : sout_(new std::ofstream()), new_line_(false), flush_(false), closed_(true), allocated_(true)
        {
        }

        explicit LogStream(std::ostream& sout, bool new_line = true, bool flush = true)
        : sout_(&sout), new_line_(new_line), flush_(flush), closed_(false), allocated_(false)
        {
        }

        ~LogStream()
        {
            if (!closed_)
            {
                close();
            }
            if (allocated_)
            {
                delete sout_;
            }
        }

        operator std::basic_ostream<char>&()
        {
            return *sout_;
        }

        template <typename T>
        LogStream& operator<<(const T& t)
        {
            if (sout_ != nullptr)
            {
                (*sout_) << t;
            }
            return *this;
        }

        LogStream& operator<<(std::basic_ostream<char>& (*func)(std::basic_ostream<char>& ))
        {
            func(*sout_);
            return *this;
        }

        void flush()
        {
            if (sout_ != nullptr)
            {
                sout_->flush();
            }
        }

        void close()
        {
            if (new_line_)
            {
                (*sout_) << std::endl;
            }
            if (flush_)
            {
                flush();
            }
            closed_ = true;
        }
    };
    
    explicit Logger()
    {}
    
    LogStream warning(bool new_line = true, bool flush = true)
    {
        LogStream stream(std::cout, new_line, flush);
        stream << "WARNING> " << prefix_;
        return stream;
    }
    
    LogStream info(bool new_line = true, bool flush = true)
    {
        LogStream stream(std::cout, new_line, flush);
        stream << "INFO> " << prefix_;
        return stream;
    }

    LogStream profile(bool new_line = true, bool flush = true)
    {
#if AIT_PROFILE || AIT_PROFILE_DISTRIBUTED
        LogStream stream(std::cout, new_line, flush);
#else
        LogStream stream;
#endif
        stream << "PROFILE> " << prefix_;
        return stream;
    }

    LogStream debug(bool new_line = true, bool flush = true)
    {
#if AIT_DEBUG
        LogStream stream(std::cout, new_line, flush);
#else
        LogStream stream;
#endif
        stream << "DEBUG> " << prefix_;
        return stream;
    }

    LogStream error(bool new_line = true, bool flush = true)
    {
        LogStream stream(std::cerr, new_line, flush);
        stream << "ERROR> " << prefix_;
        return stream;
    }

    void set_prefix(const std::string& prefix)
    {
        prefix_ = prefix;
    }
    
};

inline Logger& logger()
{
    static Logger logger_;
    return logger_;
}

inline Logger::LogStream log_warning(bool new_line = true, bool flush = true)
{
    Logger& lg = logger();
    return lg.warning(new_line, flush);
}

inline Logger::LogStream log_info(bool new_line = true, bool flush = true)
{
    Logger& lg = logger();
    return lg.info(new_line, flush);
}

inline Logger::LogStream log_profile(bool new_line = true, bool flush = true)
{
    Logger& lg = logger();
    return lg.profile(new_line, flush);
}

inline Logger::LogStream log_debug(bool new_line = true, bool flush = true)
{
    Logger& lg = logger();
    return lg.debug(new_line, flush);
}
    
inline Logger::LogStream log_error(bool new_line = true, bool flush = true)
{
    Logger& lg = logger();
    return lg.error(new_line, flush);
}

#define AIT_LOG_WARNING(expr) \
	do {\
		ait::log_warning() << expr; \
	} while (false)

#define AIT_LOG_INFO(expr) \
	do {\
		ait::log_info() << expr; \
	} while (false)

#if AIT_PROFILE || AIT_PROFILE_DISTRIBUTED
	#define AIT_LOG_PROFILE(expr) \
		do {\
			ait::log_profile() << expr; \
		} while (false)
#else
	#define AIT_LOG_PROFILE()
#endif

#if AIT_DEBUG
	#define AIT_LOG_DEBUG(expr) \
		do {\
			ait::log_debug() << expr; \
		} while (false)
#else
	#define AIT_LOG_DEBUG()
#endif

#define AIT_LOG_ERROR(expr) \
	do {\
		ait::log_error() << expr; \
	} while (false)

}
