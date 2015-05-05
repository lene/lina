/**
 * Created by lene on 02.05.15.
 */
object Tavsiye {

  type Set = Int => Boolean

  def contains(s: Set, elem: Int): Boolean = s(elem)

}
