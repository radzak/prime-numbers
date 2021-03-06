## Generowanie liczb pierwszych w Pythonie
### Czas działania programu w zależności od zakresu generowanych liczb pierwszych

1. Kod programu:
	* Do robienia pomiarów napisałem dekorator timer, który wyświetla minimalny z 5 czasów wykonania funkcji:
        ```python
        def timer(f):
            def timed(*args, **kwargs):
                exec_time = min(timeit.Timer(partial(f, *args, **kwargs)).repeat(repeat=5, number=1))
                print('%s%r took: %2.7f sec' % (f.__name__, args, exec_time))
                pass
            return timed
        ```
	* Dlaczego używam funkcji min, a nie liczę średniej ze wszystkich czasów wykonania? Otóż natknąłem się na bardzo trafną uwagę:
	<br/><br/>
	> The fastest time represents the best an algorithm can perform when the caches are loaded and the system isn't busy with other tasks. All the timings are noisy -- the fastest time is the least noisy. It is easy to show that the fastest timings are the most reproducible and therefore the most useful when timing two different implementations.
	
    * Zabawę zacząłem od najprostszej wersji algorytmu wyznaczania liczb pierwszych z zadanego przedziału <2, n> przypisywanemu Eratostenesowi z Cyreny.
        ```python
        @timer
        def eratosthenes_sieve_basic(n):
            sieve = [False, False] + [True] * (n - 1)
            for num, is_prime in enumerate(sieve):
                if is_prime:
                    for i in range(num ** 2, n + 1, num):
                        sieve[i] = False
            return list(itertools.compress(range(0, n + 1), sieve))
        ```
        
	* `list(itertools.compress(range(0, n + 1), sieve))`  
	[Funkcja compress z biblioteki itertools](https://docs.python.org/3/library/itertools.html#itertools.compress) wywołana z takimi argumentami zwraca tylko te elementy range(0, n + 1), dla których odpowiadająca im wartość w sicie jest True, w tym przypadku będą to liczby pierwsze.
    * Sprawdziłem, ile zajmuje znalezienie liczb pierwszych w zakresie od 2 do 10000000:  
    	#### eratosthenes_sieve_basic(10000000) took: 1.9464535 sec
    * Pomyślałem, że ciekawie będzie pozmieniać w tym krótkim kodzie parę rzeczy i sprawdzić jak to wpłynie na wydajność kodu. Zacząłem od zmienienia pętli, która przypisuje wielokrotnościom liczb pierwszych wartości False. Użyłem do tego wycinków (ang. slices), w Pythonie można podmienić wycinek listy inną listą. Tak więc wycinek `sieve[num ** 2:: num]` podmieniam na odpowiednią liczbę wartości False `[False] * len(sieve[num ** 2:: num])`:
	    ```python
        def eratosthenes_sieve_list_indices(n):
            sieve = [False, False] + [True] * (n - 1)
            for num, is_prime in enumerate(sieve):
                if is_prime:
                    sieve[num ** 2:: num] = [False] * len(sieve[num ** 2:: num])
            primes = list(itertools.compress(range(0, n + 1), sieve))
            return primes
        ```
		#### eratosthenes_sieve_list_indices(10000000) took: 1.8610080 sec  
        Czas nieznacznie się poprawił, jednak nie zadowoliło mnie to, także na tym nie poprzestałem.
    * Warto zauważyć, że `[False] * len(sieve[num ** 2:: num])` za każdym razem tworzy listę z odpowiednimi elementami sita. Dlaczego by nie użyć zamiast tego funkcji range(), która nie tworzy 'prawdziwej' listy?
    	```python
        def eratosthenes_sieve_list_indices_slice_proxy(n):
            sieve = [False, False] + [True] * (n - 1)
            slice_proxy = range(len(sieve))
            for num, is_prime in enumerate(sieve):
                if is_prime:
                    sieve[num ** 2:: num] = [False] * len(slice_proxy[num ** 2:: num])
            primes = list(itertools.compress(range(0, n + 1), sieve))
            return primes
        ```
		#### eratosthenes_sieve_list_indices_slice_proxy(10000000) took: 1.3616800 sec  
	* Jak widać nastąpiła duża zmiana w porównaniu do poprzedniej wersji. Zastanawiałem się jednak, czy policzenie długości tego wycinka matematycznie nie będzie szybsze.
		```python
        def eratosthenes_sieve_list_indices_mathematical(n):
            sieve = [False, False] + [True] * (n - 1)
            for num, is_prime in enumerate(sieve):
                if is_prime:
                    sieve[num ** 2:: num] = [False] * ((n - num ** 2) // num + 1)
            return list(itertools.compress(range(0, n + 1), sieve))
        ```
        #### eratosthenes_sieve_list_indices_mathematical(10000000) took: 1.2578644 sec
	* Sprawiło to, że kod stał się trochę szybszy. Byłem pewny, że da się go jeszcze jakoś zoptymalizować. Zauważyłem, że nie jest wcale potrzebne tworzenie sita na wszystkie liczby, a tylko na te nieparzyste.
		```python
        def eratosthenes_only_odd(n):
            size = n // 2
            sieve = [True] * size
            for i in range(1, int(n ** .5) + 1):
                step = 2 * i + 1
                if sieve[i]:
                    sieve[i + step::step] = [False] * (((size - 1) - i) // step)
            primes = list(itertools.compress(range(1, n + 1, 2), sieve))
            primes[0] = 2
            return primes
        ```
		#### eratosthenes_only_odd(10000000) took: 0.2595898 sec  
	* Nareszcie czas wykonania stał się zadowalający. Postanowiłem poszukać innych algorytmów i sprawdzić, czy moje ulepszone sito Eratostenesa może w jakikolwiek sposób z nimi konkurować. Benchmark 3 najszybszych algorytmów, które znalazłem i sita przedstawię na wykresie.

2. Algorytmy użyte w porównaniu:  
	* Używają one biblioteki do obliczeń numerycznych [numpy](http://www.numpy.org/).  
	`import numpy as np`
	<br/><br/>
        ```python
        @timer
        def ambi_sieve(n):
        # http://tommih.blogspot.com/2009/04/fast-prime-number-generator.html
        s = np.arange(3, n, 2)
        for m in range(3, int(n ** 0.5) + 1, 2):
            if s[(m - 3) // 2]:
                s[(m * m - 3) // 2::m] = 0
        return np.r_[2, s[s > 0]]
        ```
        ```python
        @timer
        def primesfrom3to(n):
        # https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
        """ Returns a array of primes, p < n """
            assert n >= 2
            sieve = np.ones(n // 2, dtype=np.bool)
            for i in range(3, int(n ** 0.5) + 1, 2):
                if sieve[i // 2]:
                   sieve[i * i // 2::i] = False
            return np.r_[2, 2 * np.nonzero(sieve)[0][1::] + 1]
        ```
        ```python
        @timer
        def primesfrom2to(n):
            # https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
            """ Input n>=6, Returns a array of primes, 2 <= p < n """
            sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool)
            sieve[0] = False
            for i in range(int(n ** 0.5) // 3 + 1):
                if sieve[i]:
                    k = 3 * i + 1 | 1
                    sieve[((k * k) // 3)::2 * k] = False
                    sieve[(k * k + 4 * k - 2 * k * (i & 1)) // 3::2 * k] = False
            return np.r_[2, 3, ((3 * np.nonzero(sieve)[0] + 1) | 1)]
        ```
    * Na potrzeby wykonania wykresu przerobiłem trochę dekorator timer, w taki sposób, by automatycznie przy wykonaniu funkcji najlepszy czas jej wykonania zapisywał się w słowniku `timings`:  
    `timings = defaultdict(lambda: defaultdict(list))`.
    </br></br>
        ```python
        def timer(f):
            def timed(*args, **kwargs):
                exec_time = min(timeit.Timer(partial(f, *args, **kwargs)).repeat(repeat=5, number=1))
                timings[f.__name__]['x'].append(kwargs['n'])  # x axis
                timings[f.__name__]['y'].append(exec_time)  # y axis
            return timed
        ```
    	Kluczami słownika są nazwy funkcji, a jego wartościami słowniki zawierające dane o argumentach z jakimi została wywołana funkcja i czasach jej wykonania. Sposób przechowywania tych danych może wydawać się nienajlepszy, jednak w takiej postaci w bardzo łatwy sposób będzie można utworzyć poszczególne serie wykresu.

3. Porównanie szybkości algorytmów wyznaczania liczb pierwszych.
	* Do zrobienia wykresu użyłem biblioteki matplotlib.  
	`import matplotlib.pyplot as plt`
        ```python
        def main():
            for limit in [10 ** power for power in range(3, 9)]:
                eratosthenes_only_odd(n=limit)
                ambi_sieve(n=limit)
                primesfrom3to(n=limit)
                primesfrom2to(n=limit)
        
        
        def plot():
            for name, timing in timings.items():
                plt.plot(timing['x'], timing['y'], label=name)
            
            plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%2.1f s'))
            plt.legend(loc='upper left')
            plt.show()
        
        
        if __name__ == '__main__':
            main()
            plot()
        ```  

    * Wykres zależności czasu działania programu od limitu n:
    <br/><br/>
    ![Wykres](https://github.com/radzak/prime-numbers/blob/master/wykres.png)

    |                       | 5000        | 10000       | 50000       | 100000      | 500000      | 1000000     | 5000000     | 10000000    | 50000000    | 100000000   |
    |-----------------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
    | eratosthenes_only_odd | 0.000110430 | 0.000139739 | 0.000657882 | 0.001353349 | 0.006901198 | 0.014527605 | 0.137059519 | 0.25822561  | 1.26553784  | 2.806702875 |
    | ambi_sieve            | 0.000036206 | 0.000036206 | 0.000036206 | 0.000036206 | 0.001563346 | 0.003099196 | 0.049239187 | 0.091219754 | 0.545811518 | 1.254835714 |
    | primesfrom3to         | 0.000032167 | 0.000041829 | 0.000110394 | 0.000198645 | 0.000856721 | 0.001739746 | 0.009431428 | 0.019659354 | 0.263384407 | 0.556288335 |
    | primesfrom2to         | 0.000049117 | 0.000062179 | 0.000127794 | 0.000207102 | 0.000780147 | 0.001505282 | 0.007677006 | 0.016111291 | 0.177609782 | 0.502962244 |