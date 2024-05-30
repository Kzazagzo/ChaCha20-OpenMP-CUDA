### Streszczenie

ChaCha20 jest algorytmem szyfrowania strumieniowego z kluczem symetrycznym wykonującym 20 rund operacji bitowych takie jak (XOR, permutacje, przesunięcia).

Celem projektu jest wykorzystanie mocy obliczeniowej karty graficznej do przyspieszenia procesu szyfrowania i deszyfrowania danych wykorzystując algorytm ChaCha20. Biblioteki CUDA oraz OpenMP zostaną użyte do zrównoleglenia operacji kryptograficznych, rund szyfrujących oraz operacji bitowych na wielu rdzeniach GPU.

Dane wejściowe zostaną podzielone na bloki i oddelegowane na wątki, gdzie każdy wątek będzie odpowiedzialny za szyfrowanie i odszyfrowanie pojedynczego bloku.

Kluczowe etapy działania to:

    Podział danych na bloki 64 bajtów.
    Inicjalizacja ChaCha20 dla każdego wątku GPU na podstawie klucza i wektora inicjalizującego.
    Przeprowadzenie 20 rund szyfrujących na każdym z bloków.
    Scalanie bloków w celu uzyskania ciągu wyjściowego.

### Składnia użycia skompilowanego programu
```
./ChaCha20 {metoda} {wejście} {wyjście} {klucz} {nonce}
```
Argumenty

| metoda: | cuda, openmp, cpu |
|---------|--------------------------|
| wejście: | nazwa pliku wejściowego |
| wyjście: | nazwa pliku wyjściowego |
| klucz: | opcjonalny - 8 elementowy klucz |
| nonce: | opcjonalny - 3 elementowa wartość jednorazowa |



Klucz i nonce muszą być podane po odstępach w formacie dziesiętnym jako dodatnia liczba całkowita o maksymalnym rozmiarze uint8_t. Dla ułatwienia sprawdzania te dwa argumenty są opcjonalne i przyjmują bardzo bezpieczny klucz {0,1,2,3,4,5,6,7} oraz nonce {0,1,2}.

Poprawnym działaniem programu jest zwrócenie pliku o nazwie 'wyjście' z zakodowanym ciągiem znaków. Większość popularnych błędów I/O zostały opisane specjalnymi błędami z opisem.

Mamy do czynienia z szyfrem symetrycznym, więc żeby odkodować plik wystarczy uruchomić drugi raz szyfrowanie na tym samym kluczu i wartości nonce. Podanie błędnego klucza nie odszyfruje pliku - nawet zaszyfruje go bardziej.
Kompilacja programu

Uwaga: Uruchamianie pod Windowsem nie jest wspierane, aczkolwiek możliwe - wystarczy usunąć z pliku konfiguracyjnego wpis blokujący konfigurację na nie-linuxa. Usunięcie if (UNIX) i endif() w CMakeLists.txt pozwoli na poprawną kompilację. Aczkolwiek jest to sposób nieprzetestowany, gdyż nie posiadam środowiska do kompilacji na systemach Windows.

Cała kompilacja zarządzana jest przez CMake, wymagany jest oczywiście CMake, dobry kompilator C++ (sugerowany Clang) i program do budowania (np. make). Z wymagań bibliotecznych zainstalowane narzędzia biblioteki CUDA i OpenMP (sugerowane w najnowszej wersji) są oczywiście obowiązkowe. Wszystko podłączone jest w jeden plik wykonywalny nazwany ChaCha20, którego instrukcja użycia jest powyżej.

Budowanie odbywa się przez klasyczną serię komend:
```
mkdir build
cd build
cmake ..
make
```