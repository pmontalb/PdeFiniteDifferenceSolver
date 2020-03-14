#pragma once

#include <iostream>

namespace enums
{
	template<typename T>
	class IterableEnum
	{
	public:
		class Iterator
		{
		public:
			explicit Iterator(const int iter) : _iter(iter)
			{
			}

			~Iterator() noexcept = default;
			Iterator(const Iterator&) noexcept = default;
			Iterator(Iterator&&) noexcept = default;
			Iterator& operator=(const Iterator&) noexcept = default;
			Iterator& operator=(Iterator&&) noexcept = default;

			// needed for the iteration
			T operator*() const
			{
				return static_cast<T>(_iter);
			}

			Iterator& operator++()
			{
				++_iter;
				return *this;
			}

			bool operator!=(Iterator rhs)
			{
				return _iter != rhs._iter;
			}

		private:
			int _iter;
		};
	};

	// this assumes the elements __BEGIN__ and __END__ must be defined!
	template<typename T>
	typename IterableEnum<T>::Iterator begin(IterableEnum<T>)
	{
		return typename IterableEnum<T>::Iterator(static_cast<int>(T::__BEGIN__));
	}
	template< typename T >
	typename IterableEnum<T>::Iterator end(IterableEnum<T>)
	{
		return typename IterableEnum<T>::Iterator((static_cast<int>(T::__END__)));
	}
}
