
//In Javascript, we can write comments like this, much like c++, so this language already rocks (eat it python)
//in javascript we can write to the global space like so:
var MassimoRox = 1
//This variable can be seen by any functions below. this is how we write a function

function Some_Function(A_very_normal_input) {
	
	console.log("Small_Changes: " + MassimoRox + " " + A_very_normal_input); //you're probably wondering where the this is going to log to. If you open up your html file in a browser and then click f12, you'll see a console. this is where it is logged	
}

//Now if we take a look at our html file, we made a button using bootstrap. Lets make that button do something.
//navigate back to the button code we made and add an "ID" parameter so we can reference it in the javascript code.

// lets place our button id over here so that it finds it once it is clicked
$( '#this_is_my_button_id' ).on('click', function(event) { 
  event.preventDefault(); // This line of code is used to prevent following the button link (optional)
  a_random_variable = "penoose"	//initializing a random variable
  Some_Function(a_random_variable); // Now we can call our shitty function and pass in our variable
});

//ALSO there seems to be a weird console error for bootstrap.min.js regarding 'fn', im using an older version of bootstrap and don't have this,
//but it seems to not affect anything for this version so just ignore it.