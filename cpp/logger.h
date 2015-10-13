//
//  logger.h
//  DistRandomForest
//
//  Created by Benjamin Hepp on 09/10/15.
//
//

#pragma once

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
        bool closed_;
        
    public:
        explicit LogStream(std::ostream& sout, bool new_line = true)
        : sout_(&sout), new_line_(new_line), closed_(false)
        {}

        ~LogStream()
        {
            if (!closed_)
            {
                close();
            }
        }

        operator std::basic_ostream<char>&()
        {
            return *sout_;
        }

        explicit LogStream()
        : sout_(nullptr)
        {}

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
            sout_->flush();
        }

        void close()
        {
            if (new_line_)
            {
                (*sout_) << std::endl;
            }
            flush();
            closed_ = true;
        }
    };
    
    explicit Logger()
    {}
    
    LogStream info(bool new_line = true)
    {
        LogStream stream(std::cout, new_line);
        stream << prefix_;
        return stream;
    }
    
    LogStream debug(bool new_line = true)
    {
#ifdef AIT_DEBUG
        LogStream stream(std::cout, new_line);
#else
        LogStream stream;
#endif
        stream << prefix_;
        return stream;
    }

    LogStream error(bool new_line = true)
    {
        LogStream stream(std::cerr, new_line);
        stream << prefix_;
        return stream;
    }

    void set_prefix(const std::string& prefix)
    {
        prefix_ = prefix;
    }
    
};

static Logger& logger()
{
    static Logger logger_;
    return logger_;
}

static Logger::LogStream log_info(bool new_line = true)
{
    Logger& lg = logger();
    return lg.info(new_line);
}

static Logger::LogStream log_debug(bool new_line = true)
{
    Logger& lg = logger();
    return lg.debug(new_line);
}
    
static Logger::LogStream log_error(bool new_line = true)
{
    Logger& lg = logger();
    return lg.error(new_line);
}

}