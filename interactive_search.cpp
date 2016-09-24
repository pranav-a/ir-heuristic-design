#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "meta/corpus/document.h"
#include "meta/index/inverted_index.h"
#include "meta/index/ranker/ranker_factory.h"
#include "meta/parser/analyzers/tree_analyzer.h"
#include "meta/sequence/analyzers/ngram_pos_analyzer.h"
#include "meta/util/printing.h"
#include "meta/util/time.h"

using namespace meta;

/**
 * Demo app to allow a user to create queries and search an index.
 */
std::vector<std::string> kgrams(std::string input, int len)
{
	std::vector<std::string> result;
	//This function splits the word into k-grams. Like abc to (a, ab, bc, c)
	
	if (input.length() < len)
		len = input.length();
	
	for(int i = 1; i != len; i++)
		result.push_back(input.substr(0, i)); // Add starting combinations
	
	for(int i = 0; i != input.length() - len; i++)
		result.push_back(input.substr(i, len)); // Add k-grams
	
	for(int i = len; i != 0; i --)
		result.push_back(input.substr(input.length() - i, i)); // Add ending combinations
	
	return result;
}

std::vector<std::string> cyclic(std::vector<std::string> input)
{
	int len = input.size();
	std::vector<std::string> result;
	
	// No offset
	for(int i = 1; i <= len; i++)
	{
		if(i <= (len - i + 1))
			result.push_back(std::to_string(i) + input[i - 1]); // Append ahead the numbers from start
		else
			result.push_back(input[i - 1] + std::to_string(len - i + 1)); // Append behind the numbers from end
	}
	
	// Offset of -1
	for(int i = 0; i <= len - 1; i++)
	{
		if(i <= (len - i - 1))
			result.push_back(std::to_string(i) + input[i]);
		else
			result.push_back(input[i] + std::to_string(len - i - 1));
	}
	
	//Offset of 1
	for(int i = 2; i <= len + 1; i++)
	{
		if(i <= (len - i + 3))
			result.push_back(std::to_string(i) + input[i - 2]);
		else
			result.push_back(input[i - 2] + std::to_string(len - i + 3));
	}
	return result;
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage:\t" << argv[0] << " configFile" << std::endl;
        return 1;
    }

    // Turn on logging to std::cerr.
    logging::set_cerr_logging();

    // Register additional analyzers.
    parser::register_analyzers();
    sequence::register_analyzers();

    // Create an inverted index based on the config file.
    auto config = cpptoml::parse_file(argv[1]);
    auto idx = index::make_index<index::inverted_index>(*config);

    // Create a ranking class based on the config file.
    auto group = config->get_table("ranker");
    if (!group)
        throw std::runtime_error{"\"ranker\" group needed in config file!"};
    auto ranker = index::make_ranker(*group);

    // Find the path prefix to each document so we can print out the contents.
    std::string prefix = *config->get_as<std::string>("prefix") + "/"
                         + *config->get_as<std::string>("dataset") + "/";

    std::cout << "Enter a query, or blank to quit." << std::endl << std::endl;

    std::string text;
    std::vector<std::string> result, output;
    while (true)
    {
        std::cout << "> ";
        std::getline(std::cin, text);

        if (text.empty())
            break;
		
	//Modify text string into kgrams
	result = kgrams(text, 2);
	output = cyclic(result);
	text = "";
	for(auto& kgram : output)
		text = text + kgram + " ";
		
        corpus::document query{doc_id{0}};
        query.content(text); // set the doc's content to be user input

        // Use the ranker to score the query over the index.
        std::vector<index::search_result> ranking;
        auto time = common::time([&]()
                                 {
                                     ranking = ranker->score(*idx, query, 5);
                                 });

        std::cout << "Showing top 5 results (" << time.count() << "ms)"
                  << std::endl;

        uint64_t result_num = 1;
        for (auto& result : ranking)
        {
            std::string path{idx->doc_path(result.d_id)};
            auto output
                = printing::make_bold(std::to_string(result_num) + ". " + path)
                  + " (score = " + std::to_string(result.score) + ", docid = "
                  + std::to_string(result.d_id) + ")";
            std::cout << output << std::endl;
            auto mdata = idx->metadata(result.d_id);
            if (auto content = mdata.get<std::string>("content"))
            {
                auto len
                    = std::min(std::string::size_type{77}, content->size());
                std::cout << content->substr(0, len) << "..." << std::endl
                          << std::endl;
            }
            if (result_num++ == 5)
                break;
        }
    }
}
