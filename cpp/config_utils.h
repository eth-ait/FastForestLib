//
//  config_utils.h
//  DistRandomForest
//
//  Created by Benjamin Hepp on 16/03/16.
//
//

#pragma once

#include <cstdio>
#include <rapidjson/document.h>
#include <rapidjson/filestream.h>

#include "ait.h"


namespace ait {

    class ConfigurationUtils {
    public:
        template <size_type read_buffer_size = 65536>
        static rapidjson::Document read_configuration_file(const std::string& filename) {
            std::FILE* fp = std::fopen(filename.c_str(), "rb");
            rapidjson::FileStream config_stream(fp);
            rapidjson::Document doc;
            doc.ParseStream<0>(config_stream);
            std::fclose(fp);
            return doc;
        }
        
        static bool get_bool_value(const rapidjson::Value& value, const std::string& name, bool default_value) {
            if (value.HasMember(name.c_str())) {
                assert(value[name.c_str()].IsBool_());
                return value[name.c_str()].GetBool_();
            } else {
                return default_value;
            }
        }

        static int get_int_value(const rapidjson::Value& value, const std::string& name, int default_value) {
            if (value.HasMember(name.c_str())) {
                assert(value[name.c_str()].IsInt());
                return value[name.c_str()].GetInt();
            } else {
                return default_value;
            }
        }
        
        static double get_double_value(const rapidjson::Value& value, const std::string& name, double default_value) {
            if (value.HasMember(name.c_str())) {
                assert(value[name.c_str()].IsDouble() || value[name.c_str()].IsInt());
                return value[name.c_str()].GetDouble();
            } else {
                return default_value;
            }
        }

        static std::string get_string_value(const rapidjson::Value& value, const std::string& name, const std::string& default_value) {
            if (value.HasMember(name.c_str())) {
                assert(value[name.c_str()].IsString());
                return value[name.c_str()].GetString();
            } else {
                return default_value;
            }
        }
        
        static bool get_value(const rapidjson::Value& value, const std::string& name, bool default_value) {
            return get_bool_value(value, name, default_value);
        }

        static int get_value(const rapidjson::Value& value, const std::string& name, int default_value) {
            return get_int_value(value, name, default_value);
        }
        
        static double get_value(const rapidjson::Value& value, const std::string& name, double default_value) {
            return get_double_value(value, name, default_value);
        }
        
        static std::string get_value(const rapidjson::Value& value, const std::string& name, const std::string& default_value) {
            return get_string_value(value, name, default_value);
        }
    };

}
