#pragma once

#include <iostream>
#include <cstdint>
#include <limits>
#include <boost/assert.hpp>
#include <boost/static_assert.hpp>
#include <boost/operators.hpp>
#include <boost/concept_check.hpp>
#include <cmath>

namespace numeric{
template<typename T, size_t X, size_t Y = std::numeric_limits<T>::digits - X>
class Fixed : 
  boost::ordered_field_operators<Fixed<T, X>,
  boost::unit_steppable<Fixed<T, X>, 
  boost::shiftable<Fixed<T, X>, size_t > > > {

  BOOST_STATIC_ASSERT(X < std::numeric_limits<T>::digits);
  
  class Too_large_exception: public std::exception{
    virtual const char* what() const throw()
    {
      return "Number is too large!";
    }
  } large_excep;

public:
  Fixed() {};

  template<typename TP>
  Fixed(TP x){
    data_ = static_cast<T>(x) << Y;
  }

  Fixed(double x){
    data_ = x * pow_Y + (x >= 0 ? 0.5 : -0.5);
  }
  
  Fixed(float x){
    data_ = x * pow_Y + (x >= 0 ? 0.5 : -0.5);
  }

  operator double() const{
    return data_ / pow_Y;
  }
  
  operator float() const{
    return data_ / pow_Y;
  }

  template<typename TP>
  operator TP() const{
    return data_ >> Y;
  }

  numeric::Fixed<T,X> & operator +=(const numeric::Fixed<T,X>& x){
    data_ += x.data_;
    return *this;
  }

  numeric::Fixed<T,X> & operator -=(const numeric::Fixed<T,X>& x){
    data_ -= x.data_;
    return *this;
  }

  numeric::Fixed<T,X> & operator *=(const numeric::Fixed<T,X>& x){
    data_ = static_cast<int64_t>(data_) * x.data_ >> Y;
    return *this;
  }

  numeric::Fixed<T,X> & operator /=(const numeric::Fixed<T,X>& x){
    data_ = static_cast<int64_t>(data_ << Y) / x.data_;
    return *this;
  }


  bool operator >(const numeric::Fixed<T,X>& x){
    return data_ > x.data_;
  }

  bool operator <(const numeric::Fixed<T,X>& x){
    return data_ < x.data_;
  }

  template<typename SS, typename TP, size_t XP>
  friend SS & operator<<(SS &s, const numeric::Fixed<TP,XP>& val){
    return s << (double) val;
  }

private:
  T data_;
  double pow_X = pow(2,X);
  double pow_Y = pow(2,Y);
};
} // end namespace numeric