/**
 * @brief Class implementing fixed point data type Fixed<T,X>
 * 
 * This class implements fixed point data type that uses basic arithmetic 
 * and comparison operators based on boost library implementation 
 * boost::operators<T,U>. The class implements following operators:
 *         arithmetic: +,-,*,/,++,--,-(unary)
 *         comparison: <,>,<=,>=,==,!=
 *         stream operator: << (uses conversion to double)
 * 
 * This class was influenced by the class Fixed Point Class by 
 * Peter Schregle, 2009.
 * url: https://www.codeproject.com/Articles/37636/Fixed-Point-Class
 * 
 * @author Martin Bažík
 * @date 30.04.2018
 */


#pragma once

#include <iostream>
#include <cstdint>
#include <limits>
#include <boost/assert.hpp>
#include <boost/static_assert.hpp>
#include <boost/operators.hpp>
#include <boost/concept_check.hpp>
#include <cmath>

/**
 * Namespace for the data type
 */
namespace numeric{

  /**
   * List of precalculated powers of number 2 used for fixed point conversions.
   * It contains first 63 values to fit long long data type.
   * Source: http://www.tsm-resources.com/alists/pow2.html
   */
  static constexpr long long pow2[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                      2048, 4096, 8192, 16384, 32768, 65536, 131072,
                      262144, 524288, 1048576, 2097152, 4194304,
                      8388608, 16777216, 33554432, 67108864,
                      134217728, 268435456, 536870912, 1073741824,
                      2147483648, 4294967296, 8589934592,
                      17179869184, 34359738368, 68719476736,
                      137438953472, 274877906944,549755813888,
                      1099511627776, 2199023255552, 4398046511104,
                      8796093022208, 17592186044416, 35184372088832,
                      70368744177664, 140737488355328,
                      281474976710656, 562949953421312,
                      1125899906842624, 2251799813685248,
                      4503599627370496, 9007199254740992,
                      18014398509481984, 36028797018963968,
                      72057594037927936, 144115188075855872,
                      288230376151711744, 576460752303423488,
                      1152921504606846976, 2305843009213693952,
                      4611686018427387904};


  /**
   * @class Fixed
   * @brief Template of fixed point data type
   * 
   * Parameter T represents type of variable storing data.
   * Parameter X stands for number of bits for whole part.
   * Parameter Y stands for number of bits for decimal part.
   * Parameter POW_Y  is Y-th power of 2, used for conversions.
   */
  template<typename T, size_t X, size_t Y = std::numeric_limits<T>::digits - X, const long long POW_Y = pow2[Y]>
  class Fixed : 
    boost::operators<Fixed<T, X>, size_t > {

    // There must be at least 1 bit for decimals
    BOOST_STATIC_ASSERT(X < std::numeric_limits<T>::digits);

  public:
    /**
     * Constructor
     */
    Fixed() {};

    /**
     * Conversion constructor for any type
     */
    template<typename TP>
    Fixed(TP x){
      data_ = static_cast<T>(x) << Y;
    }

    /**
     * Conversion constructor for long double
     */
    Fixed(long double x){
      data_ = x * POW_Y + (x >= 0 ? 0.5 : -0.5);
    }

    /**
     * Conversion constructor for double
     */
    Fixed(double x){
      data_ = x * POW_Y + (x >= 0 ? 0.5 : -0.5);
    }
    
    /**
     * Conversion constructor for float
     */
    Fixed(float x){
      data_ = x * POW_Y + (x >= 0 ? 0.5 : -0.5);
    }

    /**
     * Copy constructor
     */
    Fixed(const numeric::Fixed<T,X>& x){
      data_ = x.data_;
    }

    /**
     * Conversion operator to long double
     */
    operator long double() const{
      return static_cast<double>(data_) / POW_Y;
    }

    /**
     * Conversion operator to double
     */
    operator double() const{
      return static_cast<double>(data_) / POW_Y;
    }

    /**
     * Conversion operator to float
     */
    operator float() const{
      return static_cast<double>(data_) / POW_Y;
    }

    /**
     * Conversion operator to other types
     */
    template<typename TP>
    operator TP() const{
      return data_ >> Y;
    }


    /**
     * Unary - operator
     */
    numeric::Fixed<T,X> & operator -() const{
      data_ = -data_;
      return *this;
    }

    /**
     * Addition
     */
    numeric::Fixed<T,X> operator +(const numeric::Fixed<T,X>& x){
      numeric::Fixed<T,X> res;
      res.data_ = data_ + x.data_;
      return res;
    }

    /**
     * Subtraction
     */
    numeric::Fixed<T,X> operator -(const numeric::Fixed<T,X>& x){
      numeric::Fixed<T,X> res;
      res.data_ = data_ - x.data_;
      return res;
    }

    /**
     * Multiplication
     */
    numeric::Fixed<T,X> operator *(const numeric::Fixed<T,X>& x){
      numeric::Fixed<T,X> res;
      res.data_ = static_cast<int64_t>(data_) * x.data_ >> Y;
      return res;
    }

    /**
     * Division
     */
    numeric::Fixed<T,X> operator /(const numeric::Fixed<T,X>& x){
      numeric::Fixed<T,X> res;
      res.data_ = static_cast<int64_t>(data_ << Y) / x.data_;
      return res;
    }

    /**
     * Addition assignment
     */
    numeric::Fixed<T,X> & operator +=(const numeric::Fixed<T,X>& x){
      data_ += x.data_;
      return *this;
    }

    /**
     * Subtraction assignment
     */
    numeric::Fixed<T,X> & operator -=(const numeric::Fixed<T,X>& x){
      data_ -= x.data_;
      return *this;
    }

    /**
     * Multiplication assignment
     */
    numeric::Fixed<T,X> & operator *=(const numeric::Fixed<T,X>& x){
      data_ = static_cast<int64_t>(data_) * x.data_ >> Y;
      return *this;
    }

    /**
     * Division assignment
     */
    numeric::Fixed<T,X> & operator /=(const numeric::Fixed<T,X>& x){
      data_ = static_cast<int64_t>(data_ << Y) / x.data_;
      return *this;
    }
    
    /**
     * Incrementation
     */
    numeric::Fixed<T,X> & operator ++(){
      data_ += POW_Y;
      return *this;
    }
    
    /**
     * Decrementation
     */
    numeric::Fixed<T,X> & operator --(){
      data_ -= POW_Y;
      return *this;
    }
    
    /**
     * Assignment
     */
    numeric::Fixed<T,X> & operator =(const numeric::Fixed<T,X>& x){
      data_ = x.data_;
      return *this;
    }

    /**
     * Comparison operators
     */
    bool operator >(const numeric::Fixed<T,X>& x){
      return data_ > x.data_;
    }

    bool operator <(const numeric::Fixed<T,X>& x){
      return data_ < x.data_;
    }
    
    bool operator >=(const numeric::Fixed<T,X>& x){
      return data_ >= x.data_;
    }

    bool operator <=(const numeric::Fixed<T,X>& x){
      return data_ <= x.data_;
    }

    bool operator ==(const numeric::Fixed<T,X>& x){
      return data_ == x.data_;
    }

    bool operator !=(const numeric::Fixed<T,X>& x){
      return data_ != x.data_;
    }

    /**
     * Conversion operator to stream
     */
    template<typename SS, typename TP, size_t XP>
    friend SS & operator<<(SS &s, const numeric::Fixed<TP,XP>& val){
      return s << (double) val;
    }

  private:
    /**
     * Stores the data
     */
    T data_;

  }; // end class Fixed
} // end namespace numeric