//
//  convert_forest.cpp
//  DistRandomForest
//
//  Created by Benjamin Hepp on 15/03/16.
//
//

#include <cereal/archives/json.hpp>
#include <cereal/archives/binary.hpp>
#include <tclap/CmdLine.h>
#include <Eigen/Dense>

#include "ait.h"
#include "histogram_statistics.h"
#include "image_weak_learner.h"
#include "forest.h"
#include "matlab_file_io.h"

using PixelT = ait::pixel_type;
using StatisticsT = ait::HistogramStatistics;
using SplitPointT = ait::ImageSplitPoint<PixelT>;

using ForestT = ait::Forest<SplitPointT, StatisticsT>;

int main(int argc, const char* argv[]) {
    try {
        // Parse command line arguments.
        TCLAP::CmdLine cmd("Random forest converter", ' ', "0.3");
        TCLAP::ValueArg<std::string> json_forest_file_arg("j", "json-forest-file", "JSON file of the forest to load", false, "forest.json", "string");
        TCLAP::ValueArg<std::string> binary_forest_file_arg("b", "binary-forest-file", "Binary file of the forest to load", false, "forest.bin", "string");
        TCLAP::ValueArg<std::string> matlab_file_arg("m", "matlab-file", "Output .MAT-file for the converted forest", true, "", "string", cmd);
        cmd.xorAdd(json_forest_file_arg, binary_forest_file_arg);
        cmd.parse(argc, argv);

        ForestT forest;
        // Read forest from JSON file.
        if (json_forest_file_arg.isSet())
        {
            {
                ait::log_info(false) << "Reading json forest file " << json_forest_file_arg.getValue() << "... " << std::flush;
                std::ifstream ifile(json_forest_file_arg.getValue());
                cereal::JSONInputArchive iarchive(ifile);
                iarchive(cereal::make_nvp("forest", forest));
                ait::log_info(false) << " Done." << std::endl;
            }
        }
        // Read forest from binary file.
        else if (binary_forest_file_arg.isSet())
        {
            {
                ait::log_info(false) << "Reading binary forest file " << binary_forest_file_arg.getValue() << "... " << std::flush;
                std::ifstream ifile(binary_forest_file_arg.getValue(), std::ios_base::binary);
                cereal::BinaryInputArchive iarchive(ifile);
                iarchive(cereal::make_nvp("forest", forest));
                ait::log_info(false) << " Done." << std::endl;
            }
        }
        else
        {
            throw("This should never happen. Either a JSON or a binary forest file have to be specified!");
        }

        // Convert forest to array format
        // This is for image offset features with 4 offsets.
        // TODO: General interface for converting features to a vector of doubles?
        int feature_length = 4;
        int num_of_classes = forest.cbegin()->get_root_iterator()->get_statistics().num_of_bins();
        using ForestMatrixType = Eigen::MatrixXd;
        std::vector<ForestMatrixType> tree_arrays;
        for (auto tree_it = forest.cbegin(); tree_it != forest.cend(); ++tree_it) {
            const typename ForestT::TreeT& tree = *tree_it;
            ForestMatrixType tree_array(tree.size(), feature_length + 1 + num_of_classes + 1);
            for (auto node_it = tree.cbegin(); node_it != tree.cend(); ++node_it) {
                ait::size_type node_index = node_it.get_node_index();
                tree_array(node_index, 0) = node_it->get_split_point().get_offset_x1();
                tree_array(node_index, 1) = node_it->get_split_point().get_offset_y1();
                tree_array(node_index, 2) = node_it->get_split_point().get_offset_x2();
                tree_array(node_index, 3) = node_it->get_split_point().get_offset_y2();
                tree_array(node_index, feature_length) = node_it->get_split_point().get_threshold();
                ait::size_type col_offset = feature_length + 1;
                const StatisticsT& statistics = node_it->get_statistics().get_histogram();
                for (ait::size_type label = 0; label < num_of_classes; ++label) {
                    if (statistics.num_of_bins() > 0) {
                        tree_array(node_index, col_offset + label) = statistics.get_histogram()[label];
                    } else {
                        tree_array(node_index, col_offset + label) = 0;
                    }
                }
                col_offset += num_of_classes;
                if (node_it.is_leaf()) {
                    tree_array(node_index, col_offset) = 1;
                } else {
                    tree_array(node_index, col_offset) = 0;
                }
            }
            tree_arrays.push_back(std::move(tree_array));
        }

        // Write array to .MAT file
        ait::write_arrays_to_matlab_file(matlab_file_arg.getValue(), "forest", tree_arrays);
    }
    catch (const TCLAP::ArgException &e)
    {
        ait::log_error() << "Error parsing command line: " << e.error() << " for arg " << e.argId();
    }
    
    return 0;
}
