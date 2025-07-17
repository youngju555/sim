class Robot:
    def __init__(self,name,battery_level):
        self.name=name
        self.battery_level=battery_level
    def charge():
        if battery_level <= 90:
            battery_level += 10
        else :
            battery_level == 100
    def status(self):
        print(f'로봇의 이름은{self.name}이고 배터리 잔량은 {self.battery_level}입니다.')

class CleaningRobot(Robot):
    def __init__(self,name,battery_level,cleaning_mode):
        super().__init__(self,name,battery_level)
        cleaning_mode = ['nomal', 'up', 'down']
        self.cleaning_mode = cleaning_mode
    def start_cleaning(self):
        if self.battery_level < 10:
            print('배터리가 부족합니다')
        else:
            self.battery_level -= 10
        print(f'로봇의 이름은{self.name}이고 배터리 잔량은 {self.battery_level}입니다.')
        print(f'cleaning_mode : {self.cleaning_mode}입니다.') 

R1= Robot('caca',80)
R1.charge()
R1.status()
R1.start_cleaning()