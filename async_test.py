import time
import asyncio
import multiprocessing

def something():
    sum = 0
    for i in range(30000000):
        sum += i ** 2
    return sum

async def async_func(index):
    print(f'P{index} Async function is running')
    sum = something()
    print(f'P{index} Async function is done')
    return sum

def process_func(index):
    for i in range(10):
        print(f'P{index} - {i}')
        if index < 2 and i == index:
            sum = asyncio.run(async_func(index))
            print(f'P{index} Sum: {sum}')
        time.sleep(1)

if __name__ == '__main__':
    l = [None, 0, False, 1, True]
    for i in l:
        print(f"i: {i}")
        print(f"  is None: {i is None}")
        print(f"  if i: {True if i else False}")
    
    # processes = []
    # for i in range(4):
    #     p = multiprocessing.Process(target=process_func, args=(i,))
    #     processes.append(p)
    #     p.start()

    # for p in processes:
    #     p.join()