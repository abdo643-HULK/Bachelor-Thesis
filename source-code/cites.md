## Game Programming Patterns

Strategy and Type Object patterns

-   With Strategy, the goal is to decouple the main class from some portion of its behavior.

-   With Type Object, the goal is to make a number of objects behave similarly by sharing a reference to the same type object.

-   With State, the goal is for the main object to change its behavior by changing the object it delegates to.

<!-- https://docs.unity3d.com/Manual/ExecutionOrder.html -->

> As always, the golden rule of optimization is profile first. Modern computer hardware is too complex for performance to be a game of pure reason anymore.
>
> -- <quote> [Robert Nystrom][1] </quote>

> Data entities are basically maps, or property bags, or any of a dozen other terms because there’s nothing programmers like more than inventing a new name for something that already has one.
>
> We’ve re-invented them so many times that Steve Yegge calls them “The Universal Design Pattern”.
>
> -- <quote> [Robert Nystrom][1] </quote>

I worked on a game that had six million lines of C++ code. For comparison, the software controlling the Mars Curiosity rover is less than half that.

Build times for large games can vary somewhere between “go get a coffee” and “go roast your own beans, hand-grind them, pull an espresso, foam some milk, and practice your latte art in the froth”

> The “Deadly Diamond” occurs in class hierarchies with multiple inheritance where there are two different paths to the same base class. The pain that causes is a bit out of the scope of this book, but understand that they named it “deadly” for a reason.

> [1]: https://gameprogrammingpatterns.com/flyweight.html#:~:text=As%20always%2C%20the%20golden%20rule%20of%20optimization%20is%20profile%20first.%20Modern%20computer%20hardware%20is%20too%20complex%20for%20performance%20to%20be%20a%20game%20of%20pure%20reason%20anymore
> [2]: https://gameprogrammingpatterns.com/prototype.html#:~:text=We%E2%80%99ve%20re%2Dinvented%20them%20so%20many%20times%20that%20Steve%20Yegge%20calls%20them%20%E2%80%9CThe%20Universal%20Design%20Pattern%E2%80%9D.
> [3]: http://steve-yegge.blogspot.com/2008/10/universal-design-pattern.html
