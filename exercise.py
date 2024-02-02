import random
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    filename='puzzle.log',
                    encoding='utf-8', 
                    level=logging.DEBUG)

logger = logging.getLogger('puzzle_generator')


def random_exercises(args):
    if not (args.multi or args.plus or args.minus or args.custom):
        args.multi = True
        #raise Exception("No available exercise types")

    ex_types = exercise_types(args)

    # number of exercises needed
    n = args.rows * args.cols
    # list of exercises as (question, answer) tuples
    ex_list = []
    # list of answers (to avoid duplicates)
    ans_list = []

    fails = 0
    logger.debug(f"Trying to create {n} exercises")
    while len(ex_list) < n:
        q, a = random_exercise(ex_types)
        logger.debug(f"{q}={a}")
        if a not in ans_list:
            ex_list.append((q, a))
            ans_list.append(a)
        else:
            fails += 1
            logger.debug("Solution already exists")
            if fails >= n**2:
                logger.error(f"Cannot find enough unique exercises after {fails} fails.")
                raise Exception("cannot find enough unique exercises")
    logger.info(f"exercises: {ex_list}")
    return ex_list

def exercise_types(args):
    ex_types = []
    if args.multi:
        logger.debug("Using multi")
        ex_types.append(MultiExercise(args.multimaxfactor, args.multiminfactor, args.multimaxresult, args.multiminresult))
    if args.plus:
        logger.debug("Using plus")
        ex_types.append(PlusExercise(args.plusmaxsummand, args.plusminsummand, args.plusmaxresult, args.plusminresult))
    if args.minus:
        logger.debug("Using minus")
        ex_types.append(MinusExercise(args.plusmaxsummand, args.plusminsummand, args.plusmaxresult, args.plusminresult))
    if args.custom:
        logger.debug("Using custom")
        ex_types.append(CustomExercise(args.custom, args.custom_questions))
    return ex_types

def random_exercise(ex_types):
    e = random.choice(ex_types)
    return e.qa()

class Exercise:
    def __init__(self, maxval=10, minval=0, maxres=1000, minres=0):
        self.maxval = maxval
        self.minval = minval
        self.maxres = maxres
        self.minres = minres

        if minval > maxval or minres > maxres:
            raise Exception("incompatible min/max values")
    
    def qa(self):
        pass

    def get_random_operand(self, minval=None, maxval=None):
        if minval is None:
            minval = self.minval
        if maxval is None:
            maxval = self.maxval
        return random.randint(minval, maxval)

class PlusMinusExercise(Exercise):
    def ops(self):
        if self.maxval > self.maxres - self.minval:
            self.maxval = self.maxres - self.minval
        a = self.get_random_operand()
        minval = self.minval if a >= self.minres else self.minres - a
        maxval = self.maxval if self.maxres - a >= self.maxval else self.maxres - a
        b = self.get_random_operand(minval, maxval)

        c = a + b
        if c < self.minres or c > self.maxres:
            raise Exception("result error in plus")

        return a, b, c

class PlusExercise(PlusMinusExercise):
    def qa(self):
        a, b, c = self.ops()
        return f"{a}+{b}", str(c)

class MinusExercise(PlusMinusExercise):
    def qa(self):
        a, b, c = self.ops()
        return f"{c}-{a}", str(b)


class MultiExercise(Exercise):
    def qa(self):
        if self.maxval > self.maxres:
            self.maxval = self.maxres
        
        a = self.get_random_operand()
        minval = self.minval if a >= self.minres else self.minres//a + 1
        maxval = self.maxval if a == 0 or self.maxres // a >= self.maxval else self.maxres // a
        b = self.get_random_operand(minval, maxval)

        c = a * b
        if c < self.minres or c > self.maxres:
            raise Exception("result error in times")

        return f"{a}*{b}", str(c)

class RootExercise(Exercise):
    def qa(self):
        import math
        if self.maxval > self.maxres**2:
            self.maxres = int(math.sqrt(self.maxval))
        
        c = self.get_random_operand(self.minres, self.maxres)
        a = c**2

        return f"sqrt({a})", str(c)

class CustomExercise(Exercise):
    def __init__(self, fn, questions=0):
        self.questions = questions
        import pandas as pd
        logging.info(f"Reading custom exercises from {fn}")
        extension = fn.split(".")[-1]
        if extension == "xslx":
            self.df = pd.read_excel(fn, header=None)
        elif extension == "csv":
            self.df = pd.read_csv(fn, header=None)
        print(self.df)
        self.index = self.df.index.to_list()
    
    def qa(self):
        ind = random.choice(self.index)
        q, a = self.df[0][ind], self.df[1][ind]
        logger.debug(f"Randomly picked custom exercise {q}: {a} at index {ind}")
        if self.questions == 0:
            if random.random() >= 0.5:
                logger.debug("Swapping order")
                q, a = a, q
        elif self.questions == 2:
            q, a = a, q
        return q, a

        
