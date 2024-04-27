import facade
import facadeThreading
import facadeOptim


if __name__ == '__main__':
    print("Podstawowa")
    facade.facade("conf2d100.json")
    print("Dopieszczona sekwencyjna")
    facadeOptim.facade("conf2d100.json")
    print("Dopieszczona równoległa 2")
    facadeThreading.facade("conf2d100.json", 2)
    print("Dopieszczona równoległa 4")
    facadeThreading.facade("conf2d100.json", 4)
    print("Dopieszczona równoległa 8")
    facadeThreading.facade("conf2d100.json", 8)