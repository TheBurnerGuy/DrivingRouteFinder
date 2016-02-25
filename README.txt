README

Name: Henry Lo and Chris Tzanidis
Section: LBL B2 (85872)

Arduino Wiring


AdaFruit Display:

    Display Pin     Arduino Mega Pin         
    1 GND           BB GND bus
    2 Vcc           BB Vcc (+ive) bus
    3 RESET         Pin 8
    4 D/C           Pin 7
      Data/Command
    5 CARD_CS       Pin 5
      Card Chip Select 
    6 TFT_CS        Pin 6
      TFT/screen Chip Select
    7 MOSI          Pin 51
      Master Out Slave In 
    8 SCK (Clock)   Pin 52
    9 MISO          Pin 50
      Master In Slave Out 
    10 LITE         BB Vcc (+ive) bus
      Backlite

Zoom in and out buttons:

    Button          Arduino Mega Pin
    Zoom In         Pin 3
    Zoom Out        Pin 2
	or
    Modified version that debounces the button:
    Zoom In --- 560 Ohm Resistor ---|--- 1 uF capacitor --- GND
                                   |
                                   Pin 3
    Modified version that debounces the button:
    Zoom Out --- 560 Ohm Resistor ---|--- 1 uF capacitor --- GND
                                   |
                                   Pin 2

Joystick connections:
    
    Turn on pullup resistor

    Joystick Pin    Arduino Mega Pin
    Vcc             Vcc
    GND             GND
    HORZ	    Analog 0
    VER             Analog 1

    SEL             Pin 4
	or
    Modified version that debounces the button:
    SEL --- 560 Ohm Resistor ---|--- 1 uF capacitor --- GND
                                   |
                                   Pin 4

Arduino Wiring DONE

Coding instructions:
Using bash, change directory to inside the folder and run command "make upload && python3 srv.py"
