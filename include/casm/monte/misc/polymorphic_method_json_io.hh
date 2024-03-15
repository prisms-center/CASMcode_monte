#ifndef CASM_parse_polymorphic_method
#define CASM_parse_polymorphic_method

#include <functional>
#include <map>
#include <string>

#include "casm/casm_io/json/InputParser_impl.hh"
#include "casm/clexmonte/definitions.hh"

namespace CASM {

/// \brief Holds parsing functions for polymorphic implementations of BaseType
template <typename BaseType>
using MethodParserMap =
    std::map<std::string, std::function<void(InputParser<BaseType> &)>>;

/// \brief Construct implementation based on "method" and "kwargs"
///
/// This method reduces the amount of boilerplate code needed to implement
/// checking the "method" string for the name of an implementation and then
/// calling `subparse` with the "kwargs" options for each possible type.
///
/// Example:
/// - There are n different polymorphic implementations of "state_generation"
/// - The base type is `BaseType`
/// - The string `name` selects which derived type (`ImplementationType`)
///   should be constructed
/// - The input JSON looks like:
/// \code
/// {
///   "state_generation": {  // a method, of `BaseType`
///     "method": <name>,    // a name string selects the `ImplementationType`
///     "kwargs": {...}      // JSON object, options for the implementation
///   }
/// }
/// \endcode
///
/// Now, instead of writing a lot of if else checks based on the method name,
/// the BaseType parse method might just look like the following. Note that
/// different data can be forwarded to the parse method for each implementation
/// type.
/// \code
/// \brief Construct BaseType from JSON
/// void parse(
///   InputParser<BaseType> &parser,
///   DataType1 const &data1,
///   DataType2 const &data2) {
///
///   MethodParserFactory<BaseType> f;
///   parse_polymorphic_method(parser, {
///       f.make<methodA_implemention_type>("methodA")});
///       f.make<methodB_implemention_type>("methodB", data1)});
///       f.make<methodC_implemention_type>("methodC", data2)});
///       f.make<methodD_implemention_type>("methodD", data1, data2)});
/// }
/// \endcode
///
/// For the BaseType `parse` method to work, implementation type must have a
/// matching `parse` method. For this example:
/// \code
/// void parse(InputParser<methodA_implemention_type> &parser);
/// void parse(
///     InputParser<methodB_implemention_type> &parser,
///     DataType1 const &data1);
/// void parse(
///     InputParser<methodC_implemention_type> &parser,
///     DataType2 const &data2);
/// void parse(
///     InputParser<methodD_implemention_type> &parser,
///     DataType1 const &data1,
///     DataType2 const &data2);
/// \endcode
template <typename BaseType>
void parse_polymorphic_method(InputParser<BaseType> &parser,
                              MethodParserMap<BaseType> const &subparser_map) {
  // check if "method" is present
  auto json_it = parser.self.find("method");
  if (json_it == parser.self.end()) {
    parser.insert_error("method", "Missing required parameter \"method\".");
    return;
  }

  // Error method if "method" is present, but not valid
  std::stringstream method_msg;
  method_msg << "Parameter \"method\" must be one of: ";
  auto subparser_it = subparser_map.begin();
  auto subparser_end = subparser_map.end();
  while (subparser_it != subparser_end) {
    method_msg << "\"" << subparser_it->first << "\"";
    ++subparser_it;
    if (subparser_it != subparser_end) {
      method_msg << ", ";
    }
  }
  if (subparser_map.size() == 1) {
    method_msg << " (only 1 option currently)";
  }

  // "method" is not a string
  if (!json_it->is_string()) {
    parser.insert_error("method", method_msg.str());
    return;
  }

  // "method" is a string, but not a valid choice
  subparser_it = subparser_map.find(json_it->template get<std::string>());
  if (subparser_it == subparser_end) {
    parser.insert_error("method", method_msg.str());
    return;
  }

  // execute chosen parse method
  subparser_it->second(parser);
}

/// \brief Generate MethodParserMap elements
///
/// See `parse_polymorphic_method` for usage.
template <typename BaseType>
struct MethodParserFactory {
  typedef typename MethodParserMap<BaseType>::value_type return_type;

  template <typename ImplementationType, typename... Args>
  return_type make(std::string name, Args &&...args) {
    return return_type(name, [&](InputParser<BaseType> &parser) {
      std::shared_ptr<InputParser<ImplementationType>> subparser =
          parser.template subparse<ImplementationType>(
              "kwargs", std::forward<Args>(args)...);
      if (subparser->value != nullptr) {
        parser.value = std::move(subparser->value);
      }
    });
  }
};

}  // namespace CASM

#endif
