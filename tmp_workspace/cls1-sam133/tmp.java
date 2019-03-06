public class Test {
  public void setId(TAttrid node) {
    if (this._id_ != null) {
      this._id_.parent(null);
    }
    if (node != null) {
      if (node.parent() != null) {
        node.parent().removeChild(node);
      }
      node.parent(this);
    }
    this._id_ = node;
  }
}